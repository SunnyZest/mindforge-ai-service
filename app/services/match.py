"""简历-JD 匹配分析。

★ <b>结论用【五类】而不是百分比</b> —— 百分比没法行动，
  「匹配度 73%」说了等于没说；五类里每一类都直接指向一个动作：
  改表述 / 补证据 / 去学 / 问用户。

★ <b>和「简历优化」（resume.analyze_resume）的区别</b>：
  优化是「这份简历自己怎么写得更好」，不需要参照物；
  匹配是「这份简历对【这个岗位】够不够」，必须有 JD 才成立。
  所以两者不能合并 —— 输入不同、结论维度也不同。
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.services import prompt, ai_usage
from app.services.json_utils import (
    clean_dict_list,
    clean_enum,
    clean_float,
    clean_str,
    clean_str_list,
    extract_json_object,
)
from app.services.llm import llm_service

logger = logging.getLogger(__name__)

# 和抽取类任务一致：匹配是判断任务，温度要低，不要发散
_EXTRACT_TEMPERATURE = 0.1

# 输入上限
_MAX_RESUME_CHARS = 8000
_MAX_JOB_CHARS = 4000

# 逐条要求最多几条。
# ⚠️ 它直接决定输出量（每条 3 个文本字段 + 还有一份报告）。
#    实测 10 条时单次调用要 200 秒 —— 离 extract_timeout_seconds(300) 不够宽裕，
#    而模型吞吐是波动的（10~45 tok/s）。收到 8 条后有余量，JD 里
#    「锦上添花」的那两条本来也该排在后面。
#    提示词里写了同样的上限，两边要一起改。
_MAX_ITEMS = 8

# 要求属于哪一类
_CATEGORIES = {"SKILL", "EXPERIENCE", "EDUCATION", "OTHER"}

# ★ 五类结论。名字要和 Java 侧 VO 的注释、以及前端的展示分组对上。
#   顺序就是前端建议的展示顺序：先好的、后要动手的、待确认放最后。
_VERDICTS = {
    "MATCHED",                # 已匹配：简历里明确写了，能看出水平
    "EXPRESSION_GAP",         # 表达缺口：能力有但没写清 —— ★ 改一句表述就能补
    "INSUFFICIENT_EVIDENCE",  # 证据不足：提了但没有可验证的细节
    "REAL_GAP",               # 真实缺口：简历里完全没有 —— 要真去补
    "UNCERTAIN",              # 待确认：从简历判断不出来，得问用户
}

_MATCHED = "MATCHED"


async def match_resume(resume_text: str, job_text: str) -> dict[str, Any]:
    """对照 JD 逐条给出匹配结论。

    Args:
        resume_text: 简历原文
        job_text: 目标岗位 JD 原文

    Returns:
        {   # ⚠️ 键用 camelCase，理由见 _normalize 的说明
          "overallScore": float | None,   # 仅展示辅助，不参与准入
          "summary": str | None,          # 整体判断（含五类统计 + 优先动作）
          "items": [...],                 # ★ 五类结论（权威数据）
          "matchedSkills": [...], "missingSkills": [...],
          "matchedExperience": [...], "missingExperience": [...],
          "suggestions": [...],
          "truncated": bool,
        }

    Raises:
        ValueError: 简历或 JD 为空
        JsonExtractError: 模型没按格式输出
        LlmCallError: 模型调用失败
    """
    resume_clean = (resume_text or "").strip()
    job_clean = (job_text or "").strip()
    if not resume_clean:
        raise ValueError("简历文本为空，无法匹配")
    if not job_clean:
        raise ValueError("岗位 JD 为空，无法匹配")

    resume_truncated = len(resume_clean) > _MAX_RESUME_CHARS
    if resume_truncated:
        logger.warning("简历文本 %d 字符，超过上限 %d，已截断",
                       len(resume_clean), _MAX_RESUME_CHARS)
    job_truncated = len(job_clean) > _MAX_JOB_CHARS
    if job_truncated:
        logger.warning("JD 文本 %d 字符，超过上限 %d，已截断",
                       len(job_clean), _MAX_JOB_CHARS)

    system_prompt = prompt.render(
        "resume_match",
        resume_text=resume_clean[:_MAX_RESUME_CHARS],
        job_text=job_clean[:_MAX_JOB_CHARS],
    )

    raw = await llm_service.chat(
        [{"role": "system", "content": system_prompt}],
        temperature=_EXTRACT_TEMPERATURE,
        max_tokens=settings.extract_max_tokens,
        timeout=settings.extract_timeout_seconds,
        scene=ai_usage.SCENE_RESUME_MATCH,
    )

    result = _normalize(extract_json_object(raw))
    result["truncated"] = resume_truncated or job_truncated
    return result


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """规范化模型输出。

    ⚠️ 返回的键必须是 camelCase（对齐 Java 的 record / VO）——
    写成 snake_case 的话 Java 的 Jackson 对不上，字段会静默变成 null，
    接口看起来「通了」但数据全是空的（2026-10-09 踩过）。
    """
    items: list[dict[str, Any]] = []

    for src in clean_dict_list(raw.get("items"), limit=_MAX_ITEMS):
        requirement = clean_str(src.get("requirement"), max_len=300)
        if requirement is None:
            continue            # 没有要求的条目没有意义 —— 用户看不出在说哪一条

        evidence = clean_str(src.get("evidence"), max_len=500)
        verdict = clean_enum(src.get("verdict"), _VERDICTS) or "UNCERTAIN"

        # ★ 不变量：没有依据就不能说「已匹配」。
        #   模型在这里最常犯两个错：
        #     ① 把【岗位要求】当成【候选人的经历】抄进 evidence
        #        （要求「熟悉 Kafka」→ evidence 写「熟悉 Kafka」）
        #     ② 干脆不写 evidence 就判 MATCHED
        #   两种都会让用户以为自己够格、投出去被刷。所以强制降级成「待确认」——
        #   宁可让用户去确认一下，也不能给一个假的好消息。
        if verdict == _MATCHED and evidence is None:
            logger.warning("匹配结论缺少依据却判为 MATCHED，已降级为 UNCERTAIN：%s",
                           requirement[:40])
            verdict = "UNCERTAIN"

        items.append({
            "category": clean_enum(src.get("category"), _CATEGORIES) or "OTHER",
            "requirement": requirement,
            "verdict": verdict,
            "evidence": evidence,
            "advice": clean_str(src.get("advice"), max_len=400),
        })

    return {
        # 仅作展示辅助，不参与任何准入判断 —— 见提示词里的说明
        "overallScore": clean_float(raw.get("overallScore"), lo=0, hi=100),
        # ★ 整体判断（整体结论 + 五类统计 + 优先动作）。
        #   刻意【不做一份 Markdown 长报告】：五类细节已经在 items 里了，
        #   再写一份报告等于把同样的内容输出两遍 —— 实测那样要 200 秒，
        #   而离 extract_timeout_seconds(300) 就没多少余量了。
        "summary": clean_str(raw.get("summary"), max_len=600),
        # ★ items 是权威数据：前端按 verdict 分五组渲染的就是它。
        #   下面四个列表是给「快速摘要 / 表格列」用的，来自模型的直接输出
        #   （技能名要和技能表对齐，所以不能让模型写成整句要求）。
        "items": items,
        "matchedSkills": clean_str_list(raw.get("matchedSkills"),
                                        item_max_len=128, limit=30),
        "missingSkills": clean_str_list(raw.get("missingSkills"),
                                        item_max_len=128, limit=30),
        "matchedExperience": clean_str_list(raw.get("matchedExperience"),
                                            item_max_len=255, limit=20),
        "missingExperience": clean_str_list(raw.get("missingExperience"),
                                            item_max_len=255, limit=20),
        "suggestions": clean_str_list(raw.get("suggestions"),
                                      item_max_len=500, limit=15),
    }
