"""简历解析 —— 把简历文本变成结构化数据。

设计上和 rag.py / chat.py 保持一致：
  · 只吃文本、只吐结构，不碰网络、不碰数据库（纯函数）
  · 提示词放 prompts/ 里，改文案不用动代码
  · 模型输出的清洗在 json_utils 里统一做
"""

import logging
from typing import Any

from app.config import settings
from app.services import prompt
from app.services.json_utils import (
    clean_dict_list,
    clean_enum,
    clean_float,
    clean_int,
    clean_str,
    clean_str_list,
    extract_json_object,
)
from app.services.llm import llm_service

logger = logging.getLogger(__name__)

# 简历文本的输入上限。
# 一般简历也就几千字，20k 字符足够；设上限是为了防止
# 「用户上传了一本 PDF」这种极端情况把 token 烧爆。
_MAX_RESUME_CHARS = 20000

# 技能类型 / 熟练程度的合法取值，和 resume_skill 表注释对齐
_SKILL_TYPES = {"LANGUAGE", "FRAMEWORK", "TOOL", "SOFT"}
_PROFICIENCY = {"BEGINNER", "INTERMEDIATE", "ADVANCED", "EXPERT"}

# 抽取任务不需要创造力，温度调低 —— 越低越不容易「补充」原文没有的信息
_EXTRACT_TEMPERATURE = 0.1


async def parse_resume(text: str, *, target_position: str | None = None) -> dict[str, Any]:
    """把简历文本解析成结构化数据。

    Args:
        text: 简历的纯文本（Java 侧已经从 PDF/DOCX 里提取好了）
        target_position: 用户在系统里填的目标职位，可空

    Returns:
        {
          "summary": 摘要,
          "structured": {...},   # 规范化后的完整结构化简历
          "skills": [...],       # = structured["skills"]，供写 resume_skill 表
          "truncated": bool,     # 文本是否被截断过
        }

    Raises:
        ValueError: 文本为空
        JsonExtractError: 模型没按格式输出
        RuntimeError: 模型调用失败
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("简历文本为空，无法解析")

    truncated = len(cleaned) > _MAX_RESUME_CHARS
    if truncated:
        logger.warning("简历文本 %d 字符，超过上限 %d，已截断",
                       len(cleaned), _MAX_RESUME_CHARS)
        cleaned = cleaned[:_MAX_RESUME_CHARS]

    system_prompt = prompt.render(
        "resume_parse",
        resume_text=cleaned,
        target_position=target_position or "未指定",
    )

    raw = await llm_service.chat(
        [{"role": "system", "content": system_prompt}],
        temperature=_EXTRACT_TEMPERATURE,
        # ⚠️ 必须显式指定：不传的话会落到 chat_max_tokens(2000)，
        #    而这里要吐一整个简历 JSON，很容易被截断成不合法的 JSON。
        max_tokens=settings.extract_max_tokens,
        # ⚠️ 同理，超时也不能用对话的 120 秒 —— 实测详细简历要跑满 120 秒。
        timeout=settings.extract_timeout_seconds,
    )

    structured = _normalize(extract_json_object(raw))

    logger.info("简历解析完成：技能 %d 项、工作经历 %d 段、项目 %d 个%s",
                len(structured["skills"]),
                len(structured["workExperience"]),
                len(structured["projects"]),
                "（文本被截断）" if truncated else "")

    return {
        "summary": structured["summary"],
        "structured": structured,
        "skills": structured["skills"],
        "truncated": truncated,
    }


# ---------- 规范化 ----------
#
# 目的：让「模型输出」和「数据库列」对得上。
# 模型可能把年限写成 "3年"、把熟练度写成 "精通"，
# 而 resume_skill.years_experience 是 decimal(4,1)、proficiency 是枚举。
# 不洗一遍就会撞列类型，或者存进谁也认不出的值。


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": clean_str(raw.get("summary"), max_len=1000),
        "name": clean_str(raw.get("name"), max_len=64),
        "phone": clean_str(raw.get("phone"), max_len=32),
        "email": clean_str(raw.get("email"), max_len=64),
        "location": clean_str(raw.get("location"), max_len=128),
        "targetPosition": clean_str(raw.get("targetPosition"), max_len=128),
        "yearsOfExperience": clean_float(raw.get("yearsOfExperience"), hi=99.9),
        "education": _normalize_education(raw.get("education")),
        "workExperience": _normalize_work(raw.get("workExperience")),
        "projects": _normalize_projects(raw.get("projects")),
        "skills": _normalize_skills(raw.get("skills")),
        "certificates": clean_str_list(raw.get("certificates"), item_max_len=128),
        "selfEvaluation": clean_str(raw.get("selfEvaluation"), max_len=2000),
    }


def _normalize_education(value: Any) -> list[dict[str, Any]]:
    result = []
    for item in clean_dict_list(value, limit=20):
        school = clean_str(item.get("school"), max_len=128)
        if school is None:
            continue        # 连学校名都没有的条目没有意义
        result.append({
            "school": school,
            "major": clean_str(item.get("major"), max_len=128),
            "degree": clean_str(item.get("degree"), max_len=64),
            "startYear": clean_int(item.get("startYear"), lo=1900, hi=2100),
            "endYear": clean_int(item.get("endYear"), lo=1900, hi=2100),
        })
    return result


def _normalize_work(value: Any) -> list[dict[str, Any]]:
    result = []
    for item in clean_dict_list(value, limit=30):
        company = clean_str(item.get("company"), max_len=128)
        if company is None:
            continue
        result.append({
            "company": company,
            "position": clean_str(item.get("position"), max_len=128),
            "startDate": clean_str(item.get("startDate"), max_len=32),
            "endDate": clean_str(item.get("endDate"), max_len=32),
            # 提示词里要求「长原文只留关键句、不要整段照抄」，这里 1500 是安全网。
            # 之前是 4000 —— 模型会把简历原文大段搬回来，输出量翻好几倍。
            "description": clean_str(item.get("description"), max_len=1500),
        })
    return result


def _normalize_projects(value: Any) -> list[dict[str, Any]]:
    result = []
    for item in clean_dict_list(value, limit=30):
        name = clean_str(item.get("name"), max_len=128)
        if name is None:
            continue
        result.append({
            "name": name,
            "role": clean_str(item.get("role"), max_len=128),
            # 同 workExperience.description：1500 是安全网，主要靠提示词约束
            "description": clean_str(item.get("description"), max_len=1500),
            "techStack": clean_str_list(item.get("techStack"), item_max_len=64, limit=30),
        })
    return result


def _normalize_skills(value: Any) -> list[dict[str, Any]]:
    """技能列表 —— 唯一会被写进独立表的字段，所以洗得最仔细。

    ⚠️ 这里【不去重合并】：模型把「Java」和「Java 8」列成两条就保持两条。
    去重需要语义判断（Java 和 Java 8 是不是同一个？），
    猜错了会把用户真实的技能差异抹掉 —— 宁可多留一条，也不擅自合并。
    """
    result = []
    seen: set[str] = set()
    for item in clean_dict_list(value, limit=100):
        name = clean_str(item.get("name"), max_len=128)
        if name is None:
            continue
        key = name.lower()
        if key in seen:
            continue        # 只做完全同名去重，这是安全的
        seen.add(key)
        result.append({
            "name": name,
            # 白名单外的值一律置空：存「精通」进去会让后续按熟练度排序/筛选全乱
            "type": clean_enum(item.get("type"), _SKILL_TYPES),
            "proficiency": clean_enum(item.get("proficiency"), _PROFICIENCY),
            "years": clean_float(item.get("years"), hi=99.9),
        })
    return result


# ---------- 简历分析（优化建议）----------
#
# ⚠️ 和上面的 parse 是【两件事】：
#   parse   —— 从简历里【提取事实】，结果要落库
#   analyze —— 针对目标岗位给【改进建议】，结果给人看，用户自己决定采不采纳
# 所以 analyze 不落库、不覆盖任何东西。

# 建议针对简历的哪个部分，和 Java 侧 /resumes/{id}/optimize 的 target 参数对齐
_ANALYZE_TARGETS = {
    "SUMMARY", "WORK_EXPERIENCE", "PROJECT_EXPERIENCE",
    "SKILLS", "EDUCATION", "OVERALL",
}

# 最多返回几条建议。太多会让用户不知道先改哪个，
# ⚠️ 而且它直接决定输出量：每条有 5 个文本字段，8 条实测能写到 5000+ token、
#    单次调用跑满 300 秒超时。收到 5 条后典型耗时降到几十秒。
#    （提示词里也写了同样的上限和每字段字数，两边要一起改）
_MAX_ANALYZE_ITEMS = 5

# 分析时喂给模型的 JD 上限 —— 用不着整篇 JD，岗位要求通常在前 2000 字里
_MAX_JOB_CHARS = 4000


async def analyze_resume(
    text: str,
    *,
    job_text: str | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    """针对目标岗位给出简历改进建议。

    ★ <b>防虚构</b>：缺事实的地方标 `needs_fact=True`，让模型说清「要补什么」，
    而不是替用户编一个结果。约束写在 prompts/resume_analyze.txt 里，
    这里再做一道兜底（见 _normalize_analysis）。

    Args:
        text: 简历原文
        job_text: 目标 JD 原文，可空（不针对具体岗位时按通用标准评）
        target: 本次重点分析的部分，可空 = 整体

    Returns:
        {   # ⚠️ 键用 camelCase，理由见 _normalize_analysis 的说明
          "overallScore": float | None,
          "summary": str | None,
          "highlights": [...],
          "items": [...],      # 每条含 needsFact / missingFact
          "truncated": bool,
        }

    Raises:
        ValueError: 文本为空
        JsonExtractError: 模型没按格式输出
        LlmCallError: 模型调用失败
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("简历文本为空，无法分析")

    truncated = len(cleaned) > _MAX_RESUME_CHARS
    if truncated:
        logger.warning("简历文本 %d 字符，超过上限 %d，已截断",
                       len(cleaned), _MAX_RESUME_CHARS)
        cleaned = cleaned[:_MAX_RESUME_CHARS]

    job_clean = (job_text or "").strip()[:_MAX_JOB_CHARS]

    system_prompt = prompt.render(
        "resume_analyze",
        resume_text=cleaned,
        # ⚠️ 必须给个非空字符串：prompt.render 不允许占位符是空的
        job_text=job_clean or "（未提供，按通用标准评）",
        target=target or "OVERALL",
    )

    raw = await llm_service.chat(
        [{"role": "system", "content": system_prompt}],
        temperature=_EXTRACT_TEMPERATURE,
        max_tokens=settings.extract_max_tokens,
        timeout=settings.extract_timeout_seconds,
    )

    result = _normalize_analysis(extract_json_object(raw))

    logger.info("简历分析完成：%d 条建议，其中 %d 条需要用户补事实",
                len(result["items"]),
                sum(1 for i in result["items"] if i["needsFact"]))

    return {**result, "truncated": truncated}


def _normalize_analysis(raw: dict[str, Any]) -> dict[str, Any]:
    """规范化分析结果。

    ⚠️ <b>输出的键用 camelCase</b> —— 这不是笔误，也不是不 Pythonic：
    这个返回值会【原样作为 HTTP 响应】给 Java，而 Java 侧的 record 字段是 camelCase
    （`overallScore` / `needsFact`）。用 snake_case 的话 Jackson 对不上，
    那些字段会静默变成 null —— 接口看起来「通了」，但数据全是空的。
    和 resume._normalize() 的做法一致（那边也是 camelCase，对齐 Java 的 VO）。
    """
    items: list[dict[str, Any]] = []
    for item in clean_dict_list(raw.get("items"), limit=_MAX_ANALYZE_ITEMS):
        # ⚠️ 下面这些 max_len 是【安全网】，不是主要控制手段 ——
        #    主要靠提示词里写的字数上限（original ≤80 / issue ≤60 / suggestion ≤80 /
        #    rewritten ≤120 / missingFact ≤60）。这里留的余量比提示词宽 3~5 倍，
        #    正常输出不会被截，只有模型偶尔写飞了才兜一下。
        #    截断是硬切、不加省略号，所以别把这里调得比提示词还紧。
        original = clean_str(item.get("original"), max_len=500)
        suggestion = clean_str(item.get("suggestion"), max_len=400)
        # 原文和建议都拿不到的条目没有意义 —— 用户看不出在说哪一句
        if original is None and suggestion is None:
            continue

        needs_fact = _as_bool(item.get("needsFact"))
        items.append({
            "target": clean_enum(item.get("target"), _ANALYZE_TARGETS),
            "original": original,
            "issue": clean_str(item.get("issue"), max_len=300),
            "suggestion": suggestion,
            # ★ 兜底：要补事实的条目，rewritten 一律置空。
            #   模型有时候「嘴上标了 needsFact，rewritten 里却已经编好了」——
            #   那是编造的内容，绝不能漏到前端去。
            "rewritten": None if needs_fact
                         else clean_str(item.get("rewritten"), max_len=600),
            "needsFact": needs_fact,
            "missingFact": clean_str(item.get("missingFact"), max_len=300)
                           if needs_fact else None,
        })

    return {
        "overallScore": clean_float(raw.get("overallScore"), lo=0, hi=100),
        "summary": clean_str(raw.get("summary"), max_len=1000),
        "highlights": clean_str_list(raw.get("highlights"), item_max_len=255, limit=10),
        "items": items,
    }


def _as_bool(value: Any) -> bool:
    """把模型返回的「真假」统一成 bool。

    ⚠️ 不能直接 bool(value)：`bool("false")` 是 True —— 模型很爱把布尔写成字符串，
    直接用的话「不需要补事实」会被判成「需要」，建议全被吞掉。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "是"}
    if isinstance(value, (int, float)):
        return value != 0
    return False
