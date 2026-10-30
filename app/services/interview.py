"""AI 面试 —— 出题 / 答分析 / 总报告。

设计上和 resume.py / job.py 保持一致：只吃文本、只吐结构，不碰网络、不碰数据库。
提示词放 prompts/ 里，模型输出的清洗统一走 json_utils。

★ <b>为什么「一次性出全部题目」而不是每题动态生成</b>
   动态生成（根据上一题的回答决定下一题）听起来更像真人面试官，但代价是：
     · 每题多一次模型调用 —— 慢、贵，而且题量不可控（可能越问越多）
     · interview 表有 question_count、interview_question 有 question_no，
       这两个字段本身就假设了「题量固定、题号连续」
   所以这里一次性出完，逐题作答时由 Java 从已生成的题目里取下一条。
   「面试官会不会追问」这件事通过 analyze 返回的 followUp 体现 ——
   那是给用户看的（面试官可能会问什么），不改变题量。
"""

import logging
from typing import Any

from app.config import settings
from app.services import prompt, rag, ai_usage
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

# 抽取任务不需要创造力，温度调低
_EXTRACT_TEMPERATURE = 0.1

# 取值白名单 —— 和 Java 侧的常量、数据库列注释对齐
_INTERVIEW_TYPES = {"TECHNICAL", "BEHAVIORAL", "MIXED"}
_QUESTION_TYPES = {"TECHNICAL", "BEHAVIORAL", "SITUATIONAL"}
_DIFFICULTIES = {"EASY", "MEDIUM", "HARD"}

# 题量上限。再多一场面试也问不完，而且模型出到后面会开始重复
_MAX_QUESTIONS = 20

# 喂给模型的简历 / JD 上限
_MAX_RESUME_CHARS = 8000
_MAX_JOB_CHARS = 4000

# 单条回答的上限 —— 防止用户贴了一整篇文章进来把 token 烧爆
_MAX_ANSWER_CHARS = 6000

# 注入面试出题的官方智库参考资料上限
_MAX_REF_CHARS = 2500


# ---------- 出题 ----------


async def _retrieve_official_references(
    knowledge_base_id: int | str | None,
    *,
    resume_clean: str,
    job_clean: str,
    interview_type: str,
) -> str:
    """从全站官方智库（或关联知识库）检索与当前岗位/简历匹配的真题与考核标准。

    ⚠️ 增强而非强依赖：任何异常（如未配置嵌入模型、向量库为空）均静默降级为空串，
    绝不阻断面试出题主流程。
    """
    if knowledge_base_id is None or str(knowledge_base_id).strip() in ("", "0", "None", "null"):
        return ""

    query_text = f"{interview_type} {job_clean[:500]} {resume_clean[:500]}".strip()
    if not query_text:
        return ""

    try:
        results, _ = await rag.search(
            knowledge_base_id=knowledge_base_id,
            query=query_text,
            top_k=4,
            min_score=settings.min_relevance_score,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("面试出题检索官方智库失败，降级为纯简历/JD出题（kb=%s）：%s", knowledge_base_id, exc)
        return ""

    if not results:
        return ""

    pieces: list[str] = []
    total_chars = 0
    for idx, item in enumerate(results, start=1):
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if total_chars + len(content) > _MAX_REF_CHARS:
            content = content[: max(0, _MAX_REF_CHARS - total_chars)]
        if content:
            pieces.append(f"[官方智库参考 {idx}]\n{content}")
            total_chars += len(content)
        if total_chars >= _MAX_REF_CHARS:
            break

    if pieces:
        logger.info("面试出题已联合召回官方智库参考：kb=%s，命中 %d 段（共 %d 字符）",
                    knowledge_base_id, len(pieces), total_chars)
    return "\n\n".join(pieces)


async def generate_questions(
    *,
    resume_text: str | None = None,
    job_text: str | None = None,
    interview_type: str = "TECHNICAL",
    difficulty: str = "MEDIUM",
    question_count: int = 5,
    knowledge_base_id: int | str | None = None,
) -> dict[str, Any]:
    """根据简历、JD 以及（可选）官方智库真题参考出一套面试题。

    Args:
        resume_text: 候选人简历原文，可空（不基于简历面试）
        job_text: 目标岗位 JD 原文，可空
        interview_type: TECHNICAL / BEHAVIORAL / MIXED
        difficulty: EASY / MEDIUM / HARD
        question_count: 题目数量
        knowledge_base_id: 可选，官方智库（或关联知识库）ID，支持逗号分隔多 ID

    Returns:
        {"questions": [{question_no, question_type, question,
                        expected_points, difficulty}, ...]}

    Raises:
        ValueError: 简历和 JD 都没给（没东西可依据，出的题就是通用八股）
        JsonExtractError: 模型没按格式输出
        LlmCallError: 模型调用失败
    """
    resume_clean = (resume_text or "").strip()[:_MAX_RESUME_CHARS]
    job_clean = (job_text or "").strip()[:_MAX_JOB_CHARS]

    # ⚠️ 两个都不给的话，出的题必然是和简历无关的通用八股 ——
    #    那种题对用户没有价值，不如直接报错，让调用方去检查传参。
    if not resume_clean and not job_clean:
        raise ValueError("简历和 JD 至少要提供一个，否则出的题和候选人无关")

    count = max(1, min(int(question_count or 5), _MAX_QUESTIONS))

    system_prompt = prompt.render(
        "interview_generate",
        interview_type=_enum_or(interview_type, _INTERVIEW_TYPES, "TECHNICAL"),
        difficulty=_enum_or(difficulty, _DIFFICULTIES, "MEDIUM"),
        question_count=count,
        resume_text=resume_clean or "（未提供）",
        job_text=job_clean or "（未提供）",
    )

    ref_text = await _retrieve_official_references(
        knowledge_base_id,
        resume_clean=resume_clean,
        job_clean=job_clean,
        interview_type=interview_type,
    )
    if ref_text:
        system_prompt = (
            f"{system_prompt.rstrip()}\n\n"
            "【官方智库高频真题与考核标准参考】\n"
            "（以下是从平台官方智库中检索到的与该岗位/候选人技术栈高度相关的面经真题与考察标准。"
            "请在紧密贴合候选人简历与目标岗位 JD 的前提下，参考其中的深水区追问角度与评分要点来设计题目及 expectedPoints）：\n"
            f"{ref_text}\n"
        )

    raw = await llm_service.chat(
        [{"role": "system", "content": system_prompt}],
        temperature=_EXTRACT_TEMPERATURE,
        max_tokens=settings.extract_max_tokens,
        timeout=settings.extract_timeout_seconds,
        scene=ai_usage.SCENE_INTERVIEW_GENERATE,
    )

    questions = _normalize_questions(extract_json_object(raw), limit=count)

    logger.info("面试出题完成：%d 道（类型 %s，难度 %s，智库增强=%s）",
                len(questions), interview_type, difficulty, bool(ref_text))

    return {"questions": questions}


def _normalize_questions(raw: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    """规范化题目列表。

    ⚠️ 输出的键用 camelCase —— 这个返回值会原样作为 HTTP 响应给 Java，
    Java 侧的 record 字段是 camelCase。用 snake_case 的话 Jackson 对不上，
    字段会静默变成 null（接口「通了」但数据是空的）。

    ⚠️ questionNo 【由这里重排】，不信模型给的编号：
    模型经常从 0 开始、或者跳号、或者几道题共用一个号。
    Java 侧要按 questionNo 取「下一题」，编号一乱整个流程就断了。
    """
    result: list[dict[str, Any]] = []
    for item in clean_dict_list(raw.get("questions"), limit=limit):
        question = clean_str(item.get("question"), max_len=2000)
        if question is None:
            continue        # 没有题干的条目直接丢
        result.append({
            "questionNo": len(result) + 1,
            "questionType": clean_enum(item.get("questionType"), _QUESTION_TYPES),
            "question": question,
            # 参考答案要点 —— 评分时要用它对照，没有的话模型只能凭感觉打分
            "expectedPoints": clean_str_list(
                item.get("expectedPoints"), item_max_len=500, limit=10),
            "difficulty": clean_enum(item.get("difficulty"), _DIFFICULTIES),
        })
    return result


# ---------- 答分析 ----------


async def analyze_answer(
    *,
    question: str,
    answer: str,
    expected_points: list[str] | None = None,
    interview_type: str = "TECHNICAL",
) -> dict[str, Any]:
    """评估候选人对某一道题的回答。

    Args:
        question: 题目
        answer: 候选人的回答（可以是空串 —— 没作答也要给分，那就是 0 分）
        expected_points: 参考答案要点。⚠️ 传了它评分才稳 ——
            没有参照物时模型对同一份回答的打分会飘（同样的回答可能这次 70 下次 85）
        interview_type: 面试类型

    Returns:
        {   # ⚠️ 键用 camelCase，理由见 _normalize_questions 的说明
          "score": float | None,
          "technical": {"score", "strengths": [...], "weaknesses": [...]},
          "communication": {"clarity", "issues": [...]},
          "missingPoints": [...],
          "comment": str | None,
          "followUp": str | None,
        }
    """
    question_clean = (question or "").strip()
    if not question_clean:
        raise ValueError("题目为空，无法评分")

    # ⚠️ 空回答【不报错】—— 用户跳过一道题是正常行为，应该给低分而不是让整个请求失败
    answer_clean = (answer or "").strip()[:_MAX_ANSWER_CHARS]

    points_text = "；".join(
        p for p in (expected_points or []) if p
    ) or "（未提供参考答案要点，请按题目本身的合理回答标准评分）"

    system_prompt = prompt.render(
        "interview_analyze",
        interview_type=_enum_or(interview_type, _INTERVIEW_TYPES, "TECHNICAL"),
        question=question_clean,
        expected_points=points_text,
        answer=answer_clean or "（候选人未作答）",
    )

    raw = await llm_service.chat(
        [{"role": "system", "content": system_prompt}],
        temperature=_EXTRACT_TEMPERATURE,
        max_tokens=settings.extract_max_tokens,
        timeout=settings.extract_timeout_seconds,
        scene=ai_usage.SCENE_INTERVIEW_ANALYZE,
    )

    result = _normalize_analysis(extract_json_object(raw))

    logger.info("答题分析完成：得分 %s，漏了 %d 个要点",
                result["score"], len(result["missingPoints"]))

    return result


def _normalize_analysis(raw: dict[str, Any]) -> dict[str, Any]:
    """规范化单题分析。

    ⚠️ 输出的键用 camelCase —— 见 _normalize_questions 的说明。
    """
    technical = raw.get("technical") if isinstance(raw.get("technical"), dict) else {}
    communication = (
        raw.get("communication") if isinstance(raw.get("communication"), dict) else {}
    )

    return {
        "score": clean_float(raw.get("score"), lo=0, hi=100),
        "technical": {
            "score": clean_float(technical.get("score"), lo=0, hi=100),
            "strengths": clean_str_list(
                technical.get("strengths"), item_max_len=500, limit=10),
            "weaknesses": clean_str_list(
                technical.get("weaknesses"), item_max_len=500, limit=10),
        },
        "communication": {
            "clarity": clean_float(communication.get("clarity"), lo=0, hi=100),
            "issues": clean_str_list(
                communication.get("issues"), item_max_len=500, limit=10),
        },
        "missingPoints": clean_str_list(
            raw.get("missingPoints"), item_max_len=500, limit=10),
        "comment": clean_str(raw.get("comment"), max_len=1000),
        "followUp": clean_str(raw.get("followUp"), max_len=1000),
    }


# ---------- 总报告 ----------


async def build_report(
    *,
    items: list[dict[str, Any]],
    interview_type: str = "TECHNICAL",
    job_title: str | None = None,
) -> dict[str, Any]:
    """根据整场面试的问答，生成复盘报告。

    Args:
        items: 逐题记录，每项形如
            {"questionNo", "question", "answer", "score", "analysis"}
            ⚠️ score 和 analysis 必须带上 —— 报告要求「四个分数和逐题对得上」，
            模型看不到逐题分就只能瞎给，用户会发现「每题都 80 但综合给 60」
        interview_type: 面试类型
        job_title: 目标岗位名，可空

    Returns:
        {   # ⚠️ 键用 camelCase，直接对齐 interview_report 表的列
          "overallScore", "technicalScore", "communicationScore",
          "problemSolvingScore", "engineeringScore", "jobFitScore",
          "strengths": [...], "weaknesses": [...], "suggestions": [...],
          "reportText": str | None,
        }

        ⚠️ `overallScore` 是【服务端按权重算的】，不是模型给的；
           `engineeringScore` / `jobFitScore` 可能为 None（岗位未指定时后者必为 None）。

    Raises:
        ValueError: 一道题都没有
    """
    if not items:
        raise ValueError("没有任何问答记录，无法生成报告")

    system_prompt = prompt.render(
        "interview_report",
        interview_type=_enum_or(interview_type, _INTERVIEW_TYPES, "TECHNICAL"),
        job_title=(job_title or "（未指定）"),
        question_count=len(items),
        qa_dump=_format_qa_dump(items),
    )

    raw = await llm_service.chat(
        [{"role": "system", "content": system_prompt}],
        temperature=_EXTRACT_TEMPERATURE,
        max_tokens=settings.extract_max_tokens,
        timeout=settings.extract_timeout_seconds,
        scene=ai_usage.SCENE_INTERVIEW_REPORT,
    )

    result = _normalize_report(extract_json_object(raw))

    logger.info(
        "面试报告生成完成：综合 %s（技术 %s / 沟通 %s / 问题解决 %s / 工程 %s / 岗位 %s）",
        result["overallScore"],
        result["technicalScore"], result["communicationScore"],
        result["problemSolvingScore"],
        result["engineeringScore"], result["jobFitScore"])

    return result


# 报告五维加权系数。
# ⚠️ 权重只在这里定义，不要写进 prompt 让模型自己算 ——
#    带小数的加权求和模型经常算错，而且算错了看不出来：
#    你分不清是它把权重搞错了，还是它对候选人另有判断。
#
# 取值依据：公开面试评分表的常见区间（专业能力 30~40%、沟通 15~30%、
# 问题解决 15~20%、岗位匹配 5~15%），不是测算出来的 —— 模拟面试没有
# 真实结果做对照，权重在数学上无法校准，只能设、不能算。
# 用整数而不是小数，一是让人一眼看出这是设定值，二是用户能心算验一遍
# （40+20+20+10+10=100），三是整数乘加再除 100 没有浮点误差。
_REPORT_WEIGHTS: dict[str, int] = {
    "technicalScore": 40,
    "problemSolvingScore": 20,
    "communicationScore": 20,
    "engineeringScore": 10,
    "jobFitScore": 10,
}

# 模型自报总分与计算值的最大允许偏差，超过就认为模型跑偏
_OVERALL_TOLERANCE = 5.0


def _calc_overall(scores: dict[str, float | None]) -> float | None:
    """按 _REPORT_WEIGHTS 算综合分。

    ⚠️ 只对【有值】的维度加权，再把权重归一化到 1。
       缺 jobFitScore 时（没指定目标岗位）权重和只剩 90，
       不归一化会算出 70 分上下 —— 而被评估的四项其实都在 80 分以上。
       实测：技术 85 / 问题解决 80 / 沟通 76 / 工程 88，归一化 82.2，
       不归一化 74.0，差 8.2 分；而且这是系统性的，
       每一场没指定岗位的面试都会低这么多。

    ⚠️ 全部维度都为空时返回 None，不要返回 0 ——
       0 分和「没有评分」是两件事，前者会显示成一个极差的成绩。
    """
    pairs = [
        (_REPORT_WEIGHTS[key], value)
        for key, value in scores.items()
        if value is not None and key in _REPORT_WEIGHTS
    ]
    total_weight = sum(weight for weight, _ in pairs)
    if total_weight <= 0:
        return None
    return round(sum(weight * value for weight, value in pairs) / total_weight, 1)


def _normalize_report(raw: dict[str, Any]) -> dict[str, Any]:
    """规范化报告。

    ⚠️ 输出的键用 camelCase —— 见 _normalize_questions 的说明。
    这几个字段名直接对齐 `interview_report` 表的列（Java 那边要往表里写）。

    ⚠️ `overallScore` 在这里按权重算，不用模型给的 —— 见 _calc_overall。
       模型自报的那个只用来发现「它跑偏了」，会打 WARNING，但不作为最终值。
    """
    scores: dict[str, float | None] = {
        "technicalScore": clean_float(raw.get("technicalScore"), lo=0, hi=100),
        "communicationScore": clean_float(raw.get("communicationScore"), lo=0, hi=100),
        "problemSolvingScore": clean_float(
            raw.get("problemSolvingScore"), lo=0, hi=100),
        # 工程规范与岗位契合都是可空的：模型判断不了就不给
        "engineeringScore": clean_float(raw.get("engineeringScore"), lo=0, hi=100),
        "jobFitScore": clean_float(raw.get("jobFitScore"), lo=0, hi=100),
    }

    computed = _calc_overall(scores)

    # 模型自报的总分只用来发现「模型跑偏」，不作为最终值。
    # 理由：同一个「综合分」不能有时是加权、有时是模型直觉 ——
    #       那样用户没法拿它横向比较不同场次的面试。
    model_said = clean_float(raw.get("overallScore"), lo=0, hi=100)
    if model_said is not None and computed is not None:
        if abs(model_said - computed) > _OVERALL_TOLERANCE:
            logger.warning(
                "模型自报综合分 %.1f 与加权计算值 %.1f 相差超过 %.0f 分，采用计算值",
                model_said, computed, _OVERALL_TOLERANCE)

    return {
        **scores,
        "overallScore": computed,
        "strengths": clean_str_list(raw.get("strengths"), item_max_len=500, limit=10),
        "weaknesses": clean_str_list(raw.get("weaknesses"), item_max_len=500, limit=10),
        "suggestions": clean_str_list(raw.get("suggestions"), item_max_len=500, limit=10),
        "reportText": clean_str(raw.get("reportText"), max_len=20000),
    }


def _format_qa_dump(items: list[dict[str, Any]]) -> str:
    """把逐题问答拼成给模型看的文本。

    ⚠️ 必须带上【逐题得分】和【漏掉的要点】—— 报告 prompt 里要求
    「四个分数和各题得分对得上」，模型看不到逐题分就只能瞎给，
    用户会发现「每题都 80，综合却给 60」，然后不再信任这份报告。
    """
    blocks: list[str] = []
    for it in items:
        analysis = it.get("analysis") if isinstance(it.get("analysis"), dict) else {}
        technical = analysis.get("technical") if isinstance(
            analysis.get("technical"), dict) else {}

        score = it.get("score")
        answer = (it.get("answer") or "").strip()
        missing = clean_str_list(analysis.get("missingPoints"), item_max_len=200, limit=10)
        weaknesses = clean_str_list(technical.get("weaknesses"), item_max_len=200, limit=10)

        blocks.append(
            f"### 第 {it.get('questionNo')} 题"
            f"（得分 {score if score is not None else '未评分'}）\n"
            f"题目：{it.get('question') or ''}\n"
            f"回答：{answer[:_MAX_ANSWER_CHARS] or '（未作答）'}\n"
            f"漏掉的要点：{'；'.join(missing) if missing else '无'}\n"
            f"回答里的问题：{'；'.join(weaknesses) if weaknesses else '无'}"
        )
    return "\n\n".join(blocks)


# ---------- 内部工具 ----------


def _enum_or(value: Any, allowed: set[str], default: str) -> str:
    """取白名单内的值，不在就退到默认值。

    用在【喂给模型】的参数上：这些值来自 Java，正常不会错，
    但错了也不该让整个请求 500 —— 用默认值继续，比报错有用。
    """
    text = str(value or "").strip().upper()
    return text if text in allowed else default
