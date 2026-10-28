"""邀约抽取的归一与清洗。

不起服务、不调模型 —— 只测那些「模型返回什么都可能，但我们出去必须是干净形状」的地方。
重点三件事：
  ① 时间统一成北京时间、不带时区（ICS 的 Z 是 UTC，差 8 小时就发错提醒）；
  ② 枚举 / 时长越界一律置空，不猜；
  ③ evidence 只保留「字段确实有值」的那些，全空的候选直接丢掉。
"""

import asyncio
import json

import pytest

from app.services import invitation


# ---------- 时间 ----------


def test_clean_datetime_keeps_local_time_as_is():
    """不带时区的就是本地时间，不做任何换算。"""
    assert invitation._clean_datetime("2026-10-30T14:00:00") == "2026-10-30T14:00:00"
    # 分钟精度也补成完整格式，前端按 ISO 解析
    assert invitation._clean_datetime("2026-10-30T14:00") == "2026-10-30T14:00:00"


def test_clean_datetime_converts_utc_to_beijing():
    """★ ICS 里的事件时间是 UTC，必须 +8 小时。"""
    assert invitation._clean_datetime("2026-11-04T06:00:00Z") == "2026-11-04T14:00:00"


def test_clean_datetime_converts_offset():
    """其它偏移同样归到北京时间。"""
    assert invitation._clean_datetime("2026-11-04T09:00:00+09:00") == "2026-11-04T08:00:00"


def test_clean_datetime_returns_none_for_garbage():
    """解析不了就是「原文没写」，不许猜一个日期。"""
    assert invitation._clean_datetime("下周三下午两点左右") is None
    assert invitation._clean_datetime("") is None
    assert invitation._clean_datetime(None) is None


# ---------- 单条归一 ----------


def test_unknown_enum_becomes_none():
    """白名单外的值一律置空 —— 存一个前端认不出的枚举比留空更糟。"""
    event = invitation._normalize_event({
        "eventType": "TECH_ROUND",
        "eventMode": "ONSITE_MEETING",
    })
    assert event["eventType"] is None
    assert event["eventMode"] is None


def test_enum_is_case_insensitive():
    event = invitation._normalize_event({"eventType": "interview_1", "eventMode": "online"})
    assert event["eventType"] == "INTERVIEW_1"
    assert event["eventMode"] == "ONLINE"


def test_derived_is_false_when_no_time():
    """没有时间还标 derived，前端会对着空时间显示「已换算」。"""
    event = invitation._normalize_event({"eventAt": None, "eventAtDerived": True})
    assert event["eventAt"] is None
    assert event["eventAtDerived"] is False


def test_duration_is_clamped():
    assert invitation._normalize_event({"durationMinutes": 0})["durationMinutes"] is None
    assert invitation._normalize_event({"durationMinutes": 5000})["durationMinutes"] is None
    assert invitation._normalize_event({"durationMinutes": 90})["durationMinutes"] == 90


def test_matched_job_is_left_to_java():
    """职位匹配要查库，只有 Java 能做；这里必须留空，不能瞎填。"""
    event = invitation._normalize_event({"matchedJobId": "123", "matchedJobTitle": "前端"})
    assert event["matchedJobId"] is None
    assert event["matchedJobTitle"] is None


# ---------- evidence ----------


def test_evidence_dropped_for_null_fields():
    """字段是 null 就不该有 evidence —— 前端会以为是 bug。"""
    event = invitation._normalize_event({
        "companyName": "字节跳动",
        "evidence": {"companyName": "字节跳动", "eventAt": "10 月 30 日"},
    })
    assert event["evidence"] == {"companyName": "字节跳动"}


def test_evidence_dropped_for_unknown_keys():
    """只认那 8 个字段，模型多给的 key 一律丢掉。"""
    event = invitation._normalize_event({
        "title": "技术一面",
        "evidence": {"title": "技术一面", "salary": "20-30K"},
    })
    assert event["evidence"] == {"title": "技术一面"}


def test_evidence_defaults_to_empty_dict():
    """没有 evidence 时给空 dict，不是 None —— 前端类型里它是对象。"""
    event = invitation._normalize_event({"title": "技术一面"})
    assert event["evidence"] == {}


# ---------- 列表 ----------


def test_empty_event_is_dropped():
    """反例 3：只有一句「后续 HR 联系你」时，模型容易返回一条空日程。"""
    events = invitation._normalize_events([
        {"eventAt": None, "companyName": None},
        {"title": "技术一面", "eventAt": "2026-10-30T14:00:00"},
    ])
    assert len(events) == 1
    assert events[0]["title"] == "技术一面"


def test_events_are_capped():
    events = invitation._normalize_events(
        [{"title": f"第 {i} 面"} for i in range(50)]
    )
    assert len(events) == invitation._MAX_EVENTS


def test_non_list_returns_empty():
    assert invitation._normalize_events(None) == []
    assert invitation._normalize_events({"events": []}) == []


# ---------- 端到端（模型被 stub） ----------


def test_extract_end_to_end(monkeypatch):
    """跑通「原文 → 提示词 → 模型 → 归一」这条链，模型换成假的。"""
    payload = {
        "events": [{
            "title": "字节跳动 前端开发工程师 技术一面",
            "companyName": "字节跳动",
            "positionName": "前端开发工程师",
            "eventType": "INTERVIEW_1",
            "eventAt": "2026-10-30T06:00:00Z",
            "eventAtDerived": False,
            "durationMinutes": 60,
            "eventMode": "ONLINE",
            "evidence": {
                "companyName": "字节跳动",
                "eventType": "技术一面",
                "eventAt": "10 月 30 日（周四）下午 2 点",
            },
        }],
    }

    async def fake_chat(messages, **kwargs):
        return json.dumps(payload, ensure_ascii=False)

    monkeypatch.setattr(invitation.llm_service, "chat", fake_chat)

    result = asyncio.run(invitation.extract_invitation(
        text="字节跳动 前端开发工程师 技术一面，10 月 30 日（周四）下午 2 点，腾讯会议。"
    ))

    event = result["events"][0]
    assert event["companyName"] == "字节跳动"
    assert event["eventType"] == "INTERVIEW_1"
    # UTC 06:00 落到北京时间 14:00
    assert event["eventAt"] == "2026-10-30T14:00:00"
    assert event["eventAtDerived"] is False
    # 原文必须回传，Java 靠它做证据校验
    assert result["sourceText"].startswith("字节跳动")
    assert result["sourceLength"] > 0
    assert result["warning"] is None


def test_empty_text_raises():
    with pytest.raises(ValueError):
        asyncio.run(invitation.extract_invitation(text="   "))


def test_ocr_warning_is_added(monkeypatch):
    """图片输入要提示「可能有错字」—— 用户才知道要逐项核对。"""
    from app.services.parser import ParseResult

    async def fake_chat(messages, **kwargs):
        return json.dumps({"events": []}, ensure_ascii=False)

    # 图片解析换成假的：真跑 RapidOCR 要加载 ONNX 模型，单元测试里没必要
    def fake_parse(content, file_type):
        return ParseResult(text="随便一段识别出来的文字")

    monkeypatch.setattr(invitation.llm_service, "chat", fake_chat)
    monkeypatch.setattr(invitation, "parse", fake_parse)

    result = asyncio.run(invitation.extract_invitation(
        file_type="PNG", content=b"fake-image-bytes"))

    assert "OCR" in result["warning"]
    assert "没有识别出明确的日程" in result["warning"]
