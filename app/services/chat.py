"""RAG 对话：检索 → 拼 prompt → 调 LLM → 返回答案和引用来源。

和 rag.search 的区别：search 只给片段，chat 会基于片段生成一段回答。

⚠️ 权限边界和 rag.py 一样：本模块【不判权限】，
Java 已经查好权限并把确认过的 userId / knowledgeBaseId 传进来。
"""

import logging
from collections.abc import AsyncIterator

from app.config import settings
from app.services import prompt, rag, ai_usage
from app.services.llm import LlmCallError, llm_service

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

# 未检索到知识库资料时的明确免责与专业解答标识
_FALLBACK_PREFIX = "💡 **提示**：未在当前知识库中检索到直接相关资料，以下基于 MindForge AI 通用专业知识为您解答：\n\n"


async def chat(
    *,
    knowledge_base_id: int | str | None = None,
    query: str,
    top_k: int | None = None,
    history: list[dict] | None = None,
    prompt_id: str | None = None,
    enable_rerank: bool = False,
) -> dict:
    """基于知识库回答问题（或作为通用顾问直接解答）。

    Args:
        knowledge_base_id: 知识库 id。若为 None/0/空，则跳过检索直接走通用专业大模型问答。
        query: 用户问题
        top_k: 检索片段数
        history: 历史消息 [{role, content}, ...]，最多取最近 N 轮。
                 Java 侧从 chat_message 表取，不传就是单轮问答。
        prompt_id: 回答风格 id（用户在界面上选的）。不传用默认风格。

    Returns:
        {
          "answer": "...",
          "sources": [...],
          "hasContext": bool,      # 有没有检索到相关片段（≠ 有答案！）
          "hasAnswer": bool,       # 模型有没有给出实质回答
          "contextTruncated": bool # 参考资料是否因为太长被截断
        }

    Raises:
        ValueError: 参数不合法
        RuntimeError: 模型调用失败
    """
    if not query or not query.strip():
        raise ValueError("查询内容不能为空")

    # ---------- ① 检索 ----------
    is_pure_chat = (
        knowledge_base_id is None
        or str(knowledge_base_id).strip() in ("", "0", "None", "null")
    )

    if is_pure_chat:
        sources = []
    else:
        sources = await _retrieve(knowledge_base_id, query, top_k, enable_rerank)

    if not sources:
        # 如果是用户显式发起的纯通用对话（未绑定知识库），直接给出专业解答，无需附加免责提示；
        # 如果是绑定了知识库但未命中资料，附加友好的免责与来源标头。
        if is_pure_chat:
            logger.info("纯通用问答模式（无知识库绑定），直接调用大模型解答")
            prefix = ""
        else:
            logger.info("未检索到相关资料，优雅降级为通用大模型解答（kb=%s）", knowledge_base_id)
            prefix = _FALLBACK_PREFIX

        messages = _build_fallback_messages(query, history, prompt_id)
        raw_answer = await llm_service.chat(messages, scene=ai_usage.SCENE_RAG_CHAT)
        answer = f"{prefix}{raw_answer.strip()}"
        return {
            "answer": answer,
            "sources": [],
            "hasContext": False,
            "hasAnswer": True,
            "contextTruncated": False,
        }

    # ---------- ② 拼 prompt ----------
    messages, truncated = _build_messages(sources, query, history, prompt_id)

    # ---------- ③ 调模型 ----------
    answer = await llm_service.chat(messages, scene=ai_usage.SCENE_RAG_CHAT)

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


async def chat_stream(
    *,
    knowledge_base_id: int | str | None = None,
    query: str,
    top_k: int | None = None,
    history: list[dict] | None = None,
    prompt_id: str | None = None,
    enable_rerank: bool = False,
) -> AsyncIterator[dict]:
    """流式版 `chat()` —— 逐段吐事件，而不是攒完一次性返回。

    <b>产出的事件</b>（调用方按 `type` 分支）：

    ```
    {"type": "sources", "sources": [...], "hasContext": bool, "contextTruncated": bool}
    {"type": "delta",   "text": "..."}          ← 可能有很多个
    {"type": "done",    "answer": "...", "hasAnswer": bool}
    ```

    ★ <b>为什么先把 sources 推出去</b>：检索在调模型【之前】就完成了。
    先推给前端，用户能立刻看到「参考了哪几段」，不用等模型生成完 ——
    首屏感知快很多。而且它也是流的第一个字节，能让连接早点建立。

    ⚠️ <b>hasAnswer 只能等流结束才知道</b>：它靠判断完整回答里有没有那句固定话术
    （见 _NO_RESULT_MARKER），流到一半时答案还没拼完。
    所以放在 done 事件里 —— 前端可以先照常渲染流式文本，
    收到 done 再决定要不要换成「没找到」的展示。

    ⚠️ 这个生成器抛异常时，HTTP 响应头【已经发出去了】，
    调用方没法再改状态码。见 internal.py 里对异常的处理。

    Args:
        knowledge_base_id: 知识库 id。若为 None/0/空，则跳过检索直接走通用专业大模型问答。
        query: 用户问题
        top_k: 检索片段数
        history: 历史消息，最多取最近 N 轮

    Yields:
        上面三种事件（dict）

    Raises:
        ValueError: 查询为空
        LlmCallError: 模型调用失败或流中断
    """
    if not query or not query.strip():
        raise ValueError("查询内容不能为空")

    # ---------- ① 检索 ----------
    is_pure_chat = (
        knowledge_base_id is None
        or str(knowledge_base_id).strip() in ("", "0", "None", "null")
    )

    if is_pure_chat:
        sources = []
    else:
        sources = await _retrieve(knowledge_base_id, query, top_k, enable_rerank)

    if not sources:
        if is_pure_chat:
            logger.info("纯通用问答模式（无知识库绑定），流式调用大模型解答")
            prefix = ""
        else:
            logger.info("未检索到相关资料，流式降级为通用大模型解答（kb=%s）", knowledge_base_id)
            prefix = _FALLBACK_PREFIX

        yield {
            "type": "sources",
            "sources": [],
            "hasContext": False,
            "contextTruncated": False,
        }
        if prefix:
            yield {"type": "delta", "text": prefix}

        messages = _build_fallback_messages(query, history, prompt_id)
        pieces: list[str] = [prefix] if prefix else []
        async for piece in llm_service.chat_stream(messages, scene=ai_usage.SCENE_RAG_CHAT):
            pieces.append(piece)
            yield {"type": "delta", "text": piece}

        answer = "".join(pieces)
        logger.info("流式对话完成（无知识库/降级）：kb=%s，回答 %d 字（%d 段）",
                    knowledge_base_id, len(answer), len(pieces))
        yield {"type": "done", "answer": answer, "hasAnswer": True}
        return

    messages, truncated = _build_messages(sources, query, history, prompt_id)

    yield {
        "type": "sources",
        "sources": sources,
        "hasContext": True,
        "contextTruncated": truncated,
    }

    # ---------- ② 边收边吐 ----------
    # ⚠️ 必须自己把增量拼起来：hasAnswer 要拿完整回答才能判断
    pieces: list[str] = []
    async for piece in llm_service.chat_stream(messages, scene=ai_usage.SCENE_RAG_CHAT):
        pieces.append(piece)
        yield {"type": "delta", "text": piece}

    answer = "".join(pieces)
    has_answer = _NO_RESULT_MARKER not in answer

    logger.info("流式对话完成：kb=%s，参考 %d 个片段，hasAnswer=%s，回答 %d 字（%d 段）%s",
                knowledge_base_id, len(sources), has_answer, len(answer), len(pieces),
                "（参考资料被截断）" if truncated else "")

    yield {"type": "done", "answer": answer, "hasAnswer": has_answer}


# ---------- 内部 ----------


async def _retrieve(
    knowledge_base_id: int | str,
    query: str,
    top_k: int | None,
    enable_rerank: bool = False,
) -> list[dict]:
    """检索相关片段。

    `chat()` 和 `chat_stream()` 共用 —— 抽出来是为了保证两条路径
    用的是同一套检索参数（尤其 min_relevance_score，
    改一处漏一处的话，流式和非流式会给出不一样的结果）。
    """
    results, _ = await rag.search(
        knowledge_base_id=knowledge_base_id,
        query=query,
        top_k=top_k,
        min_score=settings.min_relevance_score,
        enable_rerank=enable_rerank,
    )
    return results


def _build_messages(
    sources: list[dict],
    query: str,
    history: list[dict] | None,
    prompt_id: str | None = None,
) -> tuple[list[dict], bool]:
    """把检索结果拼成 messages。返回 (messages, 参考资料是否被截断)。

    Args:
        prompt_id: 回答风格（管理端配的版本行 id）。
            不传 / 找不到 / 没开放给用户切换时，prompt_store 会回退默认那条；
            管理端一条都没配时，最终回退到 prompts/rag_chat.txt。
    """
    context, truncated = _build_context(sources)
    # 提示词优先用管理端配的（见 prompt_store），没配就回退 prompts/rag_chat.txt
    system_prompt = prompt.render("rag_chat", prompt_id=prompt_id, context=context)

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(_trim_history(history))
    messages.append({"role": "user", "content": query.strip()})
    return messages, truncated


def _build_fallback_messages(
    query: str,
    history: list[dict] | None,
    prompt_id: str | None = None,
) -> list[dict]:
    """当知识库中未检索到相关内容时，构建通用顾问降级 prompt。

    优先读取 prompts/rag_fallback.txt 或管理端下发的配置；
    若加载失败则回退内置专业求职顾问提示词。
    """
    try:
        system_prompt = prompt.render("rag_fallback", prompt_id=prompt_id)
    except Exception as exc:
        logger.warning("加载 rag_fallback 提示词失败，回退内置系统提示词: %s", exc)
        system_prompt = (
            "你是 MindForge AI 的智能求职与技术顾问。\n"
            "当前用户的专属知识库中未检索到与提问直接相关的文档资料。\n"
            "请基于你的专业通用技术知识、求职方法论与行业经验，为用户提供准确、深入、结构清晰且具有实操性的解答。\n"
            "回答规则：\n"
            "1. 用中文回答，条理清晰，重点突出，善用分点与粗体标出核心要点。\n"
            "2. 保持专业、客观、严谨，避免编造事实。"
        )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(_trim_history(history))
    messages.append({"role": "user", "content": query.strip()})
    return messages


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
