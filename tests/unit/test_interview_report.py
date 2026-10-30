"""面试报告的归一化与综合分计算。

重点只有一个：**综合分由服务端按权重算，而且必须归一化**。

为什么要专门测：
  原来综合分是模型自己加权求和，算错了看不出来（分不清是权重搞错还是它另有判断）。
  改成服务端算之后，最容易踩的坑是「只加权不归一化」——
  没指定目标岗位时 jobFitScore 为空、权重和只剩 90，
  不归一化会把每一场这样的面试都系统性压低 8 分左右（下表的第二行）。

算例取自需求文档 §4.3，已用项目里真实的 clean_float 核对过。
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.services import interview


# ---------- 综合分：五种输入 ----------


def test_overall_with_all_five_dimensions():
    scores = {
        "technicalScore": 85.0,
        "problemSolvingScore": 80.0,
        "communicationScore": 76.0,
        "engineeringScore": 88.0,
        "jobFitScore": 74.0,
    }
    # (40*85 + 20*80 + 20*76 + 10*88 + 10*74) / 100 = 81.4
    assert abs(interview._calc_overall(scores) - 81.4) < 0.05


def test_overall_without_job_fit_is_normalized():
    """★ 这条是核心：权重和只剩 90，必须归一化。

    不归一化会算出 74.0，而四项其实都在 76 分以上 —— 差 8.2 分，
    且每一场没指定岗位的面试都会低这么多。
    """
    scores = {
        "technicalScore": 85.0,
        "problemSolvingScore": 80.0,
        "communicationScore": 76.0,
        "engineeringScore": 88.0,
        "jobFitScore": None,
    }
    # (40*85 + 20*80 + 20*76 + 10*88) / 90 = 82.2
    assert abs(interview._calc_overall(scores) - 82.2) < 0.05
    # 反证：不归一化（除以 100）只有 74.0，差得离谱
    assert abs(interview._calc_overall(scores) - 74.0) > 5


def test_overall_with_only_two_dimensions():
    """维度缺得越多，不归一化错得越离谱：83.3 vs 50.0。"""
    scores = {
        "technicalScore": 85.0,
        "problemSolvingScore": 80.0,
        "communicationScore": None,
        "engineeringScore": None,
        "jobFitScore": None,
    }
    # (40*85 + 20*80) / 60 = 83.3
    assert abs(interview._calc_overall(scores) - 83.3) < 0.05


def test_overall_all_none_returns_none():
    """全空返回 None 而不是 0 —— 0 分会显示成一个极差的成绩。"""
    scores = {key: None for key in interview._REPORT_WEIGHTS}
    assert interview._calc_overall(scores) is None


def test_overall_ignores_out_of_range_dimension():
    """越界维度被 clean_float 丢掉后，权重重新归一化，不拖累整体。

    技术分 185 越界 → None，剩下四项权重和 60。
    (20*80 + 20*76 + 10*88 + 10*74) / 60 = 79.0
    """
    scores = {
        "technicalScore": None,
        "problemSolvingScore": 80.0,
        "communicationScore": 76.0,
        "engineeringScore": 88.0,
        "jobFitScore": 74.0,
    }
    assert abs(interview._calc_overall(scores) - 79.0) < 0.05


def test_overall_with_decimal_scores():
    """带一位小数的输入也要准 —— 整数权重不会引入浮点误差。"""
    scores = {
        "technicalScore": 85.5,
        "problemSolvingScore": 80.3,
        "communicationScore": 76.7,
        "engineeringScore": 88.2,
        "jobFitScore": 74.4,
    }
    assert abs(interview._calc_overall(scores) - 81.9) < 0.05


def test_weights_sum_to_one_hundred():
    """权重总和必须是 100，否则归一化的前提就不成立。"""
    assert sum(interview._REPORT_WEIGHTS.values()) == 100


# ---------- 归一化：字段与兜底 ----------


def test_normalize_report_adds_two_new_dimensions():
    result = interview._normalize_report({
        "technicalScore": 82,
        "communicationScore": 75,
        "problemSolvingScore": 76,
        "engineeringScore": 80,
        "jobFitScore": 75,
        "strengths": ["分库分表那题答得完整"],
        "reportText": "## 整体表现\n...",
    })

    assert result["engineeringScore"] == 80.0
    assert result["jobFitScore"] == 75.0
    # 综合分是算出来的，不是模型给的
    assert result["overallScore"] == pytest.approx(78.5, abs=0.05)


def test_normalize_report_keeps_missing_dimensions_as_none():
    """岗位未指定时模型返回 null → 保持 None，不能变成 0。"""
    result = interview._normalize_report({
        "technicalScore": 85,
        "communicationScore": 76,
        "problemSolvingScore": 80,
        "engineeringScore": 88,
        "jobFitScore": None,
    })

    assert result["jobFitScore"] is None
    assert result["engineeringScore"] == 88.0
    assert abs(result["overallScore"] - 82.2) < 0.05


def test_normalize_report_discards_out_of_range_score():
    result = interview._normalize_report({
        "technicalScore": 185,       # 越界，丢弃
        "communicationScore": 76,
        "problemSolvingScore": 80,
        "engineeringScore": 88,
        "jobFitScore": 74,
    })

    assert result["technicalScore"] is None
    assert abs(result["overallScore"] - 79.0) < 0.05


def test_normalize_report_prefers_computed_overall(caplog):
    """模型自报总分与计算值相差过大时，用计算值并打 WARNING。"""
    with caplog.at_level("WARNING"):
        result = interview._normalize_report({
            "technicalScore": 85,
            "communicationScore": 76,
            "problemSolvingScore": 80,
            "engineeringScore": 88,
            "jobFitScore": 74,
            "overallScore": 40,       # 模型跑偏
        })

    assert abs(result["overallScore"] - 81.4) < 0.05
    assert "相差超过" in caplog.text


def test_normalize_report_all_scores_missing():
    result = interview._normalize_report({"reportText": "..."})
    assert result["overallScore"] is None
    assert result["engineeringScore"] is None
    assert result["jobFitScore"] is None


# ---------- 整条链路：提示词 → 模型输出 → 归一化 ----------
#
# 真模型跑一次要一两分钟、还烧 token，所以日常用 mock 把「接线」测住：
# 提示词里不能再让模型算总分、模型吐的两个新维度要能进最终结果。


def _fake_model_output(**overrides):
    payload = {
        "technicalScore": 85,
        "communicationScore": 76,
        "problemSolvingScore": 80,
        "engineeringScore": 88,
        "jobFitScore": 74,
        "strengths": ["分库分表那题答得完整"],
        "weaknesses": ["慢查询那题没提回滚预案"],
        "suggestions": ["把变更回滚流程写一遍"],
        "reportText": "## 整体表现\n这几道题反映出……",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


_ITEMS = [{
    "questionNo": 1,
    "question": "说说分库分表怎么保证数据不丢",
    "answer": "双写加对账",
    "score": 85,
    "analysis": {"technical": {"score": 85, "weaknesses": []}, "missingPoints": []},
}]


def test_build_report_carries_new_dimensions_end_to_end():
    with patch("app.services.interview.llm_service.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = _fake_model_output()
        result = asyncio.run(interview.build_report(
            items=_ITEMS, interview_type="TECHNICAL", job_title="Java 后端开发工程师"))

    assert result["engineeringScore"] == 88.0
    assert result["jobFitScore"] == 74.0
    # 综合分是服务端算的：五项齐全 → 81.4
    assert abs(result["overallScore"] - 81.4) < 0.05


def test_report_prompt_tells_model_not_to_give_overall_score():
    """提示词里必须明确「总分不用你给」，否则模型给了我们也用不上，还白烧 token。"""
    with patch("app.services.interview.llm_service.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = _fake_model_output()
        asyncio.run(interview.build_report(
            items=_ITEMS, interview_type="TECHNICAL", job_title=None))

    sent = mock_chat.await_args.args[0][0]["content"]
    assert "overallScore 不用你给" in sent
    assert "engineeringScore" in sent
    assert "「（未指定）」时必须返回 null" in sent
