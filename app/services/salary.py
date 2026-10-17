"""薪酬估算：按城市 / 岗位 / 年限推算 P50 / P75 / P90 参考区间。

★ 这个模块最重要的不是「算得准」，而是**别让人以为它算得准**：
  - 我们没有真实薪酬样本，模型是在按训练数据里的岗位画像推；
  - 所以输出里刻意没有「样本量 / 数据来源」这类字段（编一个比不给更糟）；
  - `confidence` 只是模型的自我判断，Java 侧还会按城市 / 岗位 / 年限再钳一次
    （模型对什么组合都敢说 HIGH，它没有「我不确定」的动机）。

⚠️ 输出键必须 camelCase —— 对齐 Java 的 record。写成 snake_case 的话
   Jackson 对不上，字段会静默变成 null，接口看着「通了」但数据全空。
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.services import ai_usage, prompt
from app.services.json_utils import (
    clean_enum,
    clean_float,
    clean_int,
    clean_str_list,
    extract_json_object,
)
from app.services.llm import llm_service

logger = logging.getLogger(__name__)

# 估算偏判断、不需要发散，但也不像抽取那样要求零随机 —— 0.2 让区间有一点弹性
_TEMPERATURE = 0.2

# 输出很小（十几个数字 + 两三个短数组），1200 足够；给多了只是留余量
_MAX_TOKENS = 1200

_CONFIDENCE = {"LOW", "MEDIUM", "HIGH"}

# 金额上限：一亿。超过这个数只可能是模型把「万」当成了「元」
_MAX_MONEY = 100_000_000.0

# 月薪的合理下限：再低就不是薪资问题了
_MIN_MONEY = 0.0


async def estimate(
    *,
    city: str,
    position: str,
    years_of_experience: int,
    education: str | None,
    company_type: str | None,
    skills: list[str],
    current_annual_salary: float | None,
) -> dict[str, Any]:
    """推算薪酬参考区间。

    Returns:
        {   # 全部 camelCase
          "monthlyP50": float | None, "monthlyP75": ..., "monthlyP90": ...,
          "annualP50": ..., "annualP75": ..., "annualP90": ...,
          "monthsPerYear": int | None,
          "confidence": "LOW" | "MEDIUM" | "HIGH" | None,
          "basis": [...], "factors": [...],
        }
    """
    system_prompt = prompt.render(
        "salary_estimate",
        city=city,
        position=position,
        years=years_of_experience,
        education=education or "未填写",
        company_type=company_type or "未填写",
        skills="、".join(skills) if skills else "未填写",
        current_salary=_format_money(current_annual_salary),
    )

    raw = await llm_service.chat(
        [{"role": "system", "content": system_prompt}],
        temperature=_TEMPERATURE,
        max_tokens=_MAX_TOKENS,
        timeout=settings.extract_timeout_seconds,
        scene=ai_usage.SCENE_SALARY_ESTIMATE,
    )

    return _normalize(extract_json_object(raw))


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """规范化模型输出：取值域、补排序、丢掉不认识的枚举。"""
    monthly = _ascending_triple(
        "月薪",
        clean_float(raw.get("monthlyP50"), lo=_MIN_MONEY, hi=_MAX_MONEY),
        clean_float(raw.get("monthlyP75"), lo=_MIN_MONEY, hi=_MAX_MONEY),
        clean_float(raw.get("monthlyP90"), lo=_MIN_MONEY, hi=_MAX_MONEY),
    )
    annual = _ascending_triple(
        "年包",
        clean_float(raw.get("annualP50"), lo=_MIN_MONEY, hi=_MAX_MONEY),
        clean_float(raw.get("annualP75"), lo=_MIN_MONEY, hi=_MAX_MONEY),
        clean_float(raw.get("annualP90"), lo=_MIN_MONEY, hi=_MAX_MONEY),
    )

    return {
        "monthlyP50": monthly[0],
        "monthlyP75": monthly[1],
        "monthlyP90": monthly[2],
        "annualP50": annual[0],
        "annualP75": annual[1],
        "annualP90": annual[2],
        "monthsPerYear": clean_int(raw.get("monthsPerYear"), lo=12, hi=16),
        "confidence": clean_enum(raw.get("confidence"), _CONFIDENCE),
        "basis": clean_str_list(raw.get("basis"), item_max_len=200, limit=4),
        "factors": clean_str_list(raw.get("factors"), item_max_len=200, limit=4),
    }


def _ascending_triple(label: str, p50: float | None, p75: float | None,
                      p90: float | None) -> tuple[float | None, float | None, float | None]:
    """三个分位必须递增。

    模型偶尔会把顺序写反（P90 给成最低）。这里**不报错** —— 用户看到一条报错
    比看到一个修正过顺序的区间更糟 —— 但排序并留 WARN，因为这说明 prompt 或模型在漂。
    """
    values = [p50, p75, p90]
    if any(v is None for v in values):
        return p50, p75, p90
    if values[0] <= values[1] <= values[2]:
        return p50, p75, p90

    logger.warning("薪酬估算的%s分位不是递增的（%s/%s/%s），已按升序修正",
                   label, p50, p75, p90)
    ordered = sorted(values)
    return ordered[0], ordered[1], ordered[2]


def _format_money(value: float | None) -> str:
    """当前年包展示成「30 万」这种可读形式；没填就说没填。"""
    if value is None or value <= 0:
        return "未填写"
    return f"{value / 10000:.1f} 万元"
