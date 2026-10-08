"""RAG 对话：检索 → 拼 prompt → 调 LLM → 返回答案和引用来源。

和 rag.search 的区别：search 只给片段，chat 会基于片段生成一段回答。

⚠️ 权限边界和 rag.py 一样：本模块【不判权限】，
Java 已经查好权限并把确认过的 userId / knowledgeBaseId 传进来。
"""

import logging

from app.config import settings
from app.services import prompt, rag
from app.services.llm import llm_service

logger = logging.getLogger(__name__)

#   _NO_RESULT_MARKER 是【代码契约】—— hasAnswer 靠它判断模型有没有给出实质回答。
#   如果有人把它当普通文案改了措辞，判断会静默失效：不报错，
#   只是 hasAnswer 永远为 True，前端于是把「资料里没有」当成有效答案展示出来。
#
#   可改的文案（system prompt）在 prompts/rag_chat.txt；
#   这里的是逻辑依赖的字符串，改了要同步改判断逻辑。
_NO_RESULT_MARKER = "提供的资料中没有找到相关内容"

# 检索不到时直接返回给用户的兜底话术（注意开头必须包含上面的 marker）
_NO_RESULT_ANSWER = _NO_RESULT_MARKER + "请尝试换个说法，或者确认相关内容是否已上传到该知识库。"


async def chat(
    *,
    knowledge_base_id: int | str,
    query: str,
    top_k: int | None = None,
    history: list[dict] | None = None,
) -> dict:
    """基于知识库回答问题。

    Args:
        knowledge_base_id: 知识库 id —— **必传**，数据隔离的安全边界
        query: 用户问题
        top_k: 检索片段数
        history: 历史消息 [{role, content}, ...]，最多取最近 N 轮。
                 Java 侧从 chat_message 表取，不传就是单轮问答。

    Returns:
        {
          "answer": "...",
          "sources": [...],
          "hasContext": bool,      # 有没有检索到相关片段（≠ 有答案！）
          "hasAnswer": bool,       # 模型有没有给出实质回答
          "contextTruncated": bool # 参考资料是否因为太长被截断
        }

    ⚠️ hasContext 和 hasAnswer 是两回事，别混用：
        hasContext=True + hasAnswer=False 是常见情况 ——
        检索到了片段（相似度够高），但片段里并没有用户问的那个信息，
        模型如实回答「资料里没有」。这时 sources 里会有内容，
        但那些内容答不了这个问题。
        前端展示「参考了 N 个片段」用 hasContext；
        判断「要不要显示答案」用 hasAnswer。

    Raises:
        ValueError: 参数不合法
        RuntimeError: 模型调用失败
    """
    if not query or not query.strip():
        raise ValueError("查询内容不能为空")

    # ---------- ① 检索 ----------
    # min_relevance_score 过滤掉明显不相关的片段 ——
    # 宁可不答，也别拿不相关的内容硬凑，那样模型一定会编
    sources = await rag.search(
        knowledge_base_id=knowledge_base_id,
        query=query,
        top_k=top_k,
        min_score=settings.min_relevance_score,
    )

    if not sources:
        # 没检索到就【不调 LLM】—— 既省钱，也避免模型硬答
        logger.info("未检索到相关资料，直接返回兜底话术（不调用模型）")
        return {
            "answer": _NO_RESULT_ANSWER,
            "sources": [],
            "hasContext": False,
            "hasAnswer": False,
            "contextTruncated": False,
        }

    # ---------- ② 拼 prompt ----------
    context, truncated = _build_context(sources)
    # 提示词在 prompts/rag_chat.txt —— 改文案不用动代码
    system_prompt = prompt.render("rag_chat", context=context)

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(_trim_history(history))
    messages.append({"role": "user", "content": query.strip()})

    # ---------- ③ 调模型 ----------
    answer = await llm_service.chat(messages)

    # 模型按 prompt 的约定，在资料答不了时会回答那句固定话术。
    # 靠这个判断「有没有实质回答」—— 比让模型额外输出 JSON 更省 token，
    # 也比「检索到片段就算有答案」准确得多。
    has_answer = _NO_RESULT_MARKER not in answer

    logger.info("对话完成：kb=%s，参考 %d 个片段，hasAnswer=%s，回答 %d 字%s",
                knowledge_base_id, len(sources), has_answer, len(answer),
                "（参考资料被截断）" if truncated else "")

    return {
        "answer": answer,
        "sources": sources,
        "hasContext": True,        # 走到这里说明检索到了片段
        "hasAnswer": has_answer,
        "contextTruncated": truncated,
    }


# ---------- 内部 ----------


def _build_context(sources: list[dict]) -> tuple[str, bool]:
    """把检索结果拼成带编号的参考资料。

    编号是为了让模型能说「根据资料 [2]」—— 不过最终引用关系
    还是靠返回的 sources 列表，不靠模型自己说。
    """
    parts: list[str] = []
    used = 0
    truncated = False

    for i, src in enumerate(sources, start=1):
        piece = (
            f"[{i}]（来源：文档 {src.get('documentId')} 第 {src.get('chunkIndex')} 段）\n"
            f"{src.get('content', '')}"
        )
        if used + len(piece) > settings.max_context_chars:
            truncated = True
            break
        parts.append(piece)
        used += len(piece)

    return "\n\n".join(parts), truncated


def _trim_history(history: list[dict] | None) -> list[dict]:
    """只取最近 N 轮历史。

    历史太长会挤占参考资料的空间，而且对回答质量帮助有限。
    同时做一层防御：只保留 role/content 都合法的消息。
    """
    if not history:
        return []

    cleaned = [
        {"role": h.get("role"), "content": h.get("content")}
        for h in history
        if h.get("role") in ("user", "assistant") and h.get("content")
    ]

    # 一轮 = 一问一答，所以最多保留 2×N 条
    max_messages = settings.max_history_rounds * 2
    return cleaned[-max_messages:]
