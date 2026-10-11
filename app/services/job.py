"""职位 JD 解析 —— 把 JD 文本变成结构化数据。

和 resume.py 是一对：同样的模式（纯函数 + 提示词文件 + 统一清洗），
区别只在提示词和字段。
"""

import logging
from typing import Any

from app.config import settings
from app.services import prompt, ai_usage
from app.services.json_utils import (
    clean_dict_list,
    clean_enum,
    clean_str,
    clean_str_list,
    extract_json_object,
)
from app.services.llm import llm_service

logger = logging.getLogger(__name__)

# JD 通常比简历短，但也可能有长篇大论
_MAX_JOB_CHARS = 20000

_SKILL_TYPES = {"LANGUAGE", "FRAMEWORK", "TOOL", "SOFT"}

_EXTRACT_TEMPERATURE = 0.1


async def parse_job(text: str) -> dict[str, Any]:
    """把 JD 文本解析成结构化数据。

    Args:
        text: JD 原文（用户在界面上粘贴的那段）

    Returns:
        {
          "summary": 一句话概括,
          "structured": {...},   # 规范化后的完整结构化 JD
          "skills": [...],       # = structured["skills"]，供写 job_skill 表
          "truncated": bool,
        }

    Raises:
        ValueError: 文本为空
        JsonExtractError: 模型没按格式输出
        RuntimeError: 模型调用失败
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("JD 文本为空，无法解析")

    truncated = len(cleaned) > _MAX_JOB_CHARS
    if truncated:
        logger.warning("JD 文本 %d 字符，超过上限 %d，已截断",
                       len(cleaned), _MAX_JOB_CHARS)
        cleaned = cleaned[:_MAX_JOB_CHARS]

    system_prompt = prompt.render("job_parse", job_text=cleaned)

    raw = await llm_service.chat(
        [{"role": "system", "content": system_prompt}],
        temperature=_EXTRACT_TEMPERATURE,
        # ⚠️ 必须显式指定：不传会落到 chat_max_tokens(2000)，
        #    职责列表 + 技能列表很容易被截断成不合法的 JSON。
        max_tokens=settings.extract_max_tokens,
        timeout=settings.extract_timeout_seconds,
        scene=ai_usage.SCENE_JOB_PARSE,
    )

    structured = _normalize(extract_json_object(raw))

    logger.info("JD 解析完成：技能 %d 项、职责 %d 条%s",
                len(structured["skills"]),
                len(structured["responsibilities"]),
                "（文本被截断）" if truncated else "")

    return {
        "summary": structured["summary"],
        "structured": structured,
        "skills": structured["skills"],
        "truncated": truncated,
    }


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": clean_str(raw.get("summary"), max_len=500),
        "title": clean_str(raw.get("title"), max_len=255),
        "companyName": clean_str(raw.get("companyName"), max_len=255),
        "location": clean_str(raw.get("location"), max_len=255),
        # 经验/学历保留原文表述（「3-5 年」「不限」），不做标准化 ——
        # 招聘方的用词本身就是信息
        "experienceRequired": clean_str(raw.get("experienceRequired"), max_len=128),
        "educationRequired": clean_str(raw.get("educationRequired"), max_len=128),
        "responsibilities": clean_str_list(
            raw.get("responsibilities"), item_max_len=500, limit=30),
        "skills": _normalize_skills(raw.get("skills")),
        "benefits": clean_str_list(raw.get("benefits"), item_max_len=128, limit=30),
    }


def _normalize_skills(value: Any) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for item in clean_dict_list(value, limit=100):
        name = clean_str(item.get("name"), max_len=128)
        if name is None:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "name": name,
            "type": clean_enum(item.get("type"), _SKILL_TYPES),
            # requiredLevel 保留原话（「精通」「熟悉」）——
            # 换算是丢信息，而且 job_skill.required_level 就是 varchar
            "requiredLevel": clean_str(item.get("requiredLevel"), max_len=32),
            # required 是布尔语义：没写就按「加分项」处理（false），
            # 因为 JD 里没提「必须」的，多半就是可选项
            "required": bool(item.get("required", False)),
        })
    return result
