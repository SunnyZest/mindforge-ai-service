"""面试邀约抽取 —— 把 HR 发来的邀约原文抽成「日程候选」。

三种输入：粘贴文本（微信邀面词 / 邮件正文）、PDF / DOCX、PNG / JPG 截图。
文件先过 `parser.parse()` 转文本，再交给模型。

设计上的三个要点：

1. **只抽不推**。抽不到就 null，禁止默认值 —— 由提示词约束，Java 侧再用
   evidence 子串校验兜一道（模型编的东西在原文里找不到依据，会被丢掉）。
   这一层刻意**不**替模型补默认值：补了就等于把幻觉洗成事实。

2. **不落库**。返回的是候选，用户在前端逐条确认后才由 Java 写 `job_event`。

3. **时间只允许「相对表述换算」这一种推导**，且必须回 `eventAtDerived=true`，
   前端标黄让人核对。原文写死日期的必须是 false。
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import settings
from app.services import ai_usage, prompt
from app.services.json_utils import (
    clean_bool,
    clean_enum,
    clean_int,
    clean_str,
    extract_json_object,
)
from app.services.llm import llm_service
from app.services.parser import parse

logger = logging.getLogger(__name__)

# 日程类型白名单，与 Java 侧 JobEventSaveDTO 的正则、前端 JobEventType 完全一致。
# 三处任何一处改动都要同步 —— 不然用户在前端看到的值后端存不进去。
_EVENT_TYPES = {"WRITTEN_TEST", "INTERVIEW_1", "INTERVIEW_2", "HR", "OFFER_TALK", "OTHER"}

# 面试形式白名单，同上。
_EVENT_MODES = {"ONSITE", "ONLINE", "PHONE"}

# 允许出现在 evidence 里的字段，与前端 ExtractedEventField 一一对应。
# ⚠️ 顺序也是「从强到弱」：出问题时看日志里哪个字段的证据最容易丢，就知道该调哪段提示词。
_EVIDENCE_FIELDS = (
    "title",
    "companyName",
    "positionName",
    "eventType",
    "eventAt",
    "durationMinutes",
    "eventMode",
    "location",
)

# 单次最多接受的候选条数。一封邀约撑死三五轮，超过就是模型把每句话都当成一条日程了。
_MAX_EVENTS = 20

# 温度固定 0：这是抽取任务，不需要任何「创造性」。
_TEMPERATURE = 0.0

# 北京时间。
#
# ⚠️ 这里刻意**不用** ZoneInfo("Asia/Shanghai")：Windows 上没有系统时区数据库，
#    必须额外装 `tzdata` 包才能用，漏装就是启动期 ZoneInfoNotFoundError ——
#    而开发和测试恰恰在 Windows 上跑。
#    中国自 1991 年起不再使用夏令时，UTC+8 固定偏移和 Asia/Shanghai 完全等价，
#    所以直接用固定偏移，省掉一个只在 Windows 上才炸的依赖。
_CST = timezone(timedelta(hours=8))

_WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

# 走 OCR 的类型 —— 用来决定要不要提示「识别可能有错字」
_OCR_TYPES = {"PNG", "JPG", "JPEG"}


async def extract_invitation(
    *,
    text: str | None = None,
    file_type: str | None = None,
    content: bytes | None = None,
) -> dict[str, Any]:
    """从邀约原文里抽日程候选。

    Args:
        text: 粘贴的原文（与 content 二选一）
        file_type: 文件类型，如 PDF / DOCX / PNG / JPG
        content: 文件字节（与 text 二选一）

    Returns:
        {
          "sourceText": 解析后的正文,   # ⚠️ Java 靠它做证据校验，必须回传
          "events": [...],
          "sourceLength": 有效字符数,
          "warning": str | None,
          "truncated": bool,
        }

    Raises:
        ValueError: 既没给文本也没给文件 / 文本为空
        UnsupportedFileTypeError: 文件类型不支持
        ParseError: 文件读不出文字（加密、损坏、图片里没字）
        LlmCallError: 模型调用失败
        JsonExtractError: 模型没按格式输出
    """
    source_text, ocr_used = await _resolve_source(
        text=text, file_type=file_type, content=content)

    truncated = len(source_text) > settings.invitation_max_chars
    if truncated:
        logger.warning("邀约原文 %d 字符，超过上限 %d，已截断",
                       len(source_text), settings.invitation_max_chars)
        source_text = source_text[:settings.invitation_max_chars]

    now = datetime.now(_CST)
    system_prompt = prompt.render(
        "invitation_extract",
        invitation_text=source_text,
        today=now.strftime("%Y-%m-%d"),
        weekday=_WEEKDAYS[now.weekday()],
    )

    raw = await llm_service.chat(
        [{"role": "system", "content": system_prompt}],
        temperature=_TEMPERATURE,
        # ⚠️ 必须显式传：不传会落到 chat_max_tokens(2000)，虽然也够，但超时会落到
        #    客户端默认的 120 秒 —— 那是给对话定的，邀约用不着等那么久。
        max_tokens=settings.invitation_max_tokens,
        timeout=settings.invitation_timeout_seconds,
        scene=ai_usage.SCENE_INVITATION_EXTRACT,
        # ★ 关掉「思考」。2026-10-28 实测：开着的时候，一封含两轮面试的邀约邮件
        #   跑满 60 秒超时直接失败（重试也超），单轮微信邀约要 50 秒；
        #   关掉后同样的输入 10 秒内出结果，字段和证据都没变。
        #   这里是「照抄原文里的日程」，不需要先推理一遍。
        enable_thinking=settings.extract_enable_thinking,
    )

    events = _normalize_events(extract_json_object(raw).get("events"))

    logger.info("邀约抽取完成：原文 %d 字符 → %d 条候选%s",
                len(source_text), len(events),
                "（原文被截断）" if truncated else "")

    return {
        "sourceText": source_text,
        "events": events,
        "sourceLength": _count_chars(source_text),
        "warning": _build_warning(ocr_used=ocr_used, truncated=truncated, event_count=len(events)),
        "truncated": truncated,
    }


# ---------- 输入 ----------


async def _resolve_source(
    *, text: str | None, file_type: str | None, content: bytes | None
) -> tuple[str, bool]:
    """拿到「要分析的正文」，并告诉调用方这份正文是不是 OCR 出来的。

    ⚠️ 文件输入时**必须把提取出来的正文回传**（见返回值的 sourceText）：
    Java 侧没有解析器，手上没有原文就没法做 evidence 子串校验，
    整条防幻觉的链路会直接失效（那边会返回 50201）。
    """
    if content is not None:
        normalized_type = (file_type or "").strip().upper()
        # ⚠️ 必须让出线程：图片 OCR 要 1~3 秒，扫描件 PDF 更久。
        #    卡在事件循环里的话，这一个请求会把整个服务（含别人的对话）拖死。
        result = await asyncio.to_thread(parse, content, normalized_type)
        return result.text, normalized_type in _OCR_TYPES

    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("邀约内容为空，无法识别")
    return cleaned, False


def _count_chars(text: str) -> int:
    """有效字符数（不含空白），与 Java 侧算法一致。"""
    return sum(1 for ch in text if not ch.isspace())


# ---------- 输出归一 ----------


def _normalize_events(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    events: list[dict[str, Any]] = []
    for item in value[:_MAX_EVENTS]:
        if not isinstance(item, dict):
            continue
        event = _normalize_event(item)
        # 全空的一条是模型「凑数」的产物（反例 3 那种）。
        # 丢掉它，别让用户在预览表里看到一行什么都没有的日程。
        if _is_empty(event):
            logger.debug("邀约抽取返回了一条全空日程，已丢弃")
            continue
        events.append(event)

    if isinstance(value, list) and len(value) > _MAX_EVENTS:
        logger.warning("邀约抽取返回 %d 条候选，只保留前 %d 条", len(value), _MAX_EVENTS)

    return events


def _normalize_event(raw: dict[str, Any]) -> dict[str, Any]:
    event = {
        "title": clean_str(raw.get("title"), max_len=255),
        "companyName": clean_str(raw.get("companyName"), max_len=255),
        "positionName": clean_str(raw.get("positionName"), max_len=255),
        "eventType": clean_enum(raw.get("eventType"), _EVENT_TYPES),
        "eventAt": _clean_datetime(raw.get("eventAt")),
        "eventAtDerived": clean_bool(raw.get("eventAtDerived"), default=False),
        "durationMinutes": clean_int(raw.get("durationMinutes"), lo=1, hi=1440),
        "eventMode": clean_enum(raw.get("eventMode"), _EVENT_MODES),
        "location": clean_str(raw.get("location"), max_len=500),
        "meetingUrl": clean_str(raw.get("meetingUrl"), max_len=500),
        "interviewer": clean_str(raw.get("interviewer"), max_len=100),
        "note": clean_str(raw.get("note"), max_len=500),
        # 职位匹配在 Java 侧做（它才连数据库），这里占位成 null 保持形状一致
        "matchedJobId": None,
        "matchedJobTitle": None,
    }

    # 时间没抽到时 derived 必须是 false —— 否则前端会对着空时间显示「已换算」
    if event["eventAt"] is None:
        event["eventAtDerived"] = False

    event["evidence"] = _normalize_evidence(raw.get("evidence"), event)
    return event


def _clean_datetime(value: Any) -> str | None:
    """把模型给的时间统一成 `YYYY-MM-DDTHH:mm:ss`（北京时间、不带时区）。

    ⚠️ 带 `Z` / `+09:00` 这类偏移的输入会**换算到北京时间**再输出，不是丢掉偏移量。
    ICS 日历里的时间是 UTC，直接当本地时间用会整整差 8 小时 ——
    表现是邮件提醒提前 8 小时发出去，而用户完全不知道是时区问题。
    （Java 侧也做了同样的兜底，两边都做是因为这条错误的代价太高、又极难发现。）
    """
    text = clean_str(value)
    if text is None:
        return None

    # Python 3.11+ 的 fromisoformat 认 "Z"；低版本要自己换掉
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        logger.debug("邀约里的时间不是合法 ISO 格式，按「原文没写」处理：%s", text)
        return None

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(_CST).replace(tzinfo=None)

    return parsed.strftime("%Y-%m-%dT%H:%M:%S")


def _normalize_evidence(value: Any, event: dict[str, Any]) -> dict[str, str]:
    """只保留「白名单字段 + 该字段确实有值」的证据片段。

    ⚠️ 字段是 null 就不该有 evidence —— 前端拿到一条「有证据没值」的记录会以为出了 bug，
       而且 Java 侧的证据闸也是按「有值才校验」的逻辑走的。
    """
    if not isinstance(value, dict):
        return {}

    result: dict[str, str] = {}
    for field in _EVIDENCE_FIELDS:
        if event.get(field) is None:
            continue
        snippet = clean_str(value.get(field), max_len=500)
        if snippet:
            result[field] = snippet
    return result


def _is_empty(event: dict[str, Any]) -> bool:
    """除了证据和占位字段之外，一个有效信息都没有。"""
    return all(
        event.get(field) is None
        for field in ("title", "companyName", "positionName", "eventType",
                      "eventAt", "durationMinutes", "eventMode", "location",
                      "meetingUrl", "interviewer", "note")
    )


def _build_warning(*, ocr_used: bool, truncated: bool, event_count: int) -> str | None:
    """拼提示文案。

    只写「模型和解析器自己知道的事」—— 被证据闸丢了哪些字段，只有 Java 知道，
    它会把自己那份拼在后面（见 InvitationExtractServiceImpl）。
    """
    warnings: list[str] = []
    if truncated:
        warnings.append("原文较长，只分析了前面一部分，后面可能有遗漏")
    if ocr_used:
        warnings.append("图片是文字识别（OCR）出来的，可能有错字，请逐项核对")
    if event_count == 0:
        warnings.append("没有识别出明确的日程 —— 原文里可能没有具体时间，建议手动新建")

    return "；".join(warnings) if warnings else None
