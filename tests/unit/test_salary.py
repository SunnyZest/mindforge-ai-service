"""薪酬估算的规范化与防御。

这个模块的「准不准」测不了（没有真实样本），但**能不能守住边界**可以测：
分位顺序、值域、置信度枚举、月数范围 —— 模型输出漂了的时候，
这些兜底决定了用户看到的是「一个偏保守的区间」还是「一条离谱的数字」。
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.services import salary


def model_reply(**overrides) -> str:
    """一份正常的模型输出。"""
    payload = {
        "monthlyP50": 25000,
        "monthlyP75": 32000,
        "monthlyP90": 42000,
        "annualP50": 350000,
        "annualP75": 448000,
        "annualP90": 588000,
        "monthsPerYear": 14,
        "confidence": "HIGH",
        "basis": ["一线城市 + 3 年经验", "Java 后端供给充足"],
        "factors": ["掌握 Flink 可上浮"],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


async def run(reply: str) -> dict:
    with patch("app.services.salary.llm_service.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = reply
        return await salary.estimate(
            city="杭州",
            position="Java 后端开发",
            years_of_experience=3,
            education="本科",
            company_type="大厂",
            skills=["Java", "MySQL"],
            current_annual_salary=300000,
        )


@pytest.mark.anyio
async def test_normal_output_is_camel_case():
    data = await run(model_reply())

    assert set(data) == {
        "monthlyP50", "monthlyP75", "monthlyP90",
        "annualP50", "annualP75", "annualP90",
        "monthsPerYear", "confidence", "basis", "factors",
    }
    assert data["monthlyP50"] == 25000
    assert data["monthsPerYear"] == 14
    assert data["confidence"] == "HIGH"
    assert data["basis"] == ["一线城市 + 3 年经验", "Java 后端供给充足"]


@pytest.mark.anyio
async def test_reversed_percentiles_are_sorted():
    """模型把 P90 写成最低时，按升序修正而不是原样返回。"""
    data = await run(model_reply(monthlyP50=42000, monthlyP75=32000, monthlyP90=25000))

    assert data["monthlyP50"] == 25000
    assert data["monthlyP75"] == 32000
    assert data["monthlyP90"] == 42000


@pytest.mark.anyio
async def test_unknown_confidence_is_dropped():
    """不在枚举里的置信度一律置空 —— 宁可没有，也不要传一个前端不认识的字符串。"""
    data = await run(model_reply(confidence="VERY_HIGH"))
    assert data["confidence"] is None


@pytest.mark.anyio
async def test_out_of_range_months_is_dropped():
    """月数只可能是 12~16。写 24 或 3 都是模型把「月薪」和「年薪」搞混了。"""
    assert (await run(model_reply(monthsPerYear=24)))["monthsPerYear"] is None
    assert (await run(model_reply(monthsPerYear=3)))["monthsPerYear"] is None


@pytest.mark.anyio
async def test_absurd_money_is_dropped():
    """超过一亿只可能是把「万」当成了「元」，不能原样透给用户。"""
    data = await run(model_reply(monthlyP50=300000000, monthlyP75=320000000,
                                 monthlyP90=340000000))
    assert data["monthlyP50"] is None


@pytest.mark.anyio
async def test_missing_percentile_stays_none():
    """缺一个分位就保持 None —— 由 Java 侧统一判定「结果不完整」，
    这里不擅自用别的数补齐（补出来的数字没人能负责）。"""
    data = await run(model_reply(monthlyP75=None))
    assert data["monthlyP50"] == 25000
    assert data["monthlyP75"] is None


@pytest.mark.anyio
async def test_basis_and_factors_are_capped():
    """依据最多 4 条、每条 200 字 —— 模型很容易写成长段落。

    另外 clean_str_list 会去重，所以这里用 10 条互不相同的依据。
    """
    data = await run(model_reply(
        basis=[f"依据 {i}" for i in range(10)],
        factors=["因素" * 300]))

    assert len(data["basis"]) == 4
    assert len(data["factors"][0]) <= 200


@pytest.mark.anyio
async def test_prompt_receives_joined_skills():
    """技能要拼成一段话交给 prompt，空列表给「未填写」而不是空串。"""
    with patch("app.services.salary.prompt.render") as mock_render, \
         patch("app.services.salary.llm_service.chat", new_callable=AsyncMock) as mock_chat:
        mock_render.return_value = "prompt"
        mock_chat.return_value = model_reply()

        await salary.estimate(
            city="杭州", position="Java 后端", years_of_experience=3,
            education=None, company_type=None, skills=["Java", "Redis"],
            current_annual_salary=None,
        )

        kwargs = mock_render.call_args.kwargs
        assert kwargs["skills"] == "Java、Redis"
        assert kwargs["education"] == "未填写"
        assert kwargs["company_type"] == "未填写"
        assert kwargs["current_salary"] == "未填写"


def test_format_money():
    assert salary._format_money(None) == "未填写"
    assert salary._format_money(0) == "未填写"
    assert salary._format_money(300000) == "30.0 万元"
