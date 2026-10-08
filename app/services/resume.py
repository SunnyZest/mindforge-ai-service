"""简历解析 —— 把简历文本变成结构化数据。

设计上和 rag.py / chat.py 保持一致：
  · 只吃文本、只吐结构，不碰网络、不碰数据库（纯函数）
  · 提示词放 prompts/ 里，改文案不用动代码
  · 模型输出的清洗在 json_utils 里统一做
"""

import logging
from typing import Any

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
            "description": clean_str(item.get("description"), max_len=4000),
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
            "description": clean_str(item.get("description"), max_len=4000),
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
