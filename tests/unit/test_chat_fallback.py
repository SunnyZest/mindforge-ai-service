"""RAG 对话无私域资料时优雅降级单元测试。"""

from unittest.mock import AsyncMock, patch
import pytest

from app.services import chat
from app.services.chat import _FALLBACK_PREFIX


@pytest.mark.anyio
async def test_chat_fallback_when_no_sources():
    """未检索到知识库资料时：优雅降级为通用大模型专业解答，带免责/来源标头。"""
    with patch("app.services.chat._retrieve", new_callable=AsyncMock) as mock_retrieve, \
         patch("app.services.chat.llm_service.chat", new_callable=AsyncMock) as mock_llm_chat:
        mock_retrieve.return_value = []
        mock_llm_chat.return_value = "Java 内存模型（JMM）主要定义了线程和主内存之间的抽象关系。"

        result = await chat.chat(
            knowledge_base_id="kb-test",
            query="Java 内存模型有哪些？",
        )

        assert result["hasContext"] is False
        assert result["hasAnswer"] is True
        assert result["sources"] == []
        assert result["contextTruncated"] is False
        assert result["answer"].startswith(_FALLBACK_PREFIX)
        assert "Java 内存模型（JMM）" in result["answer"]
        mock_llm_chat.assert_called_once()


@pytest.mark.anyio
async def test_chat_stream_fallback_when_no_sources():
    """流式对话未命中资料时：先推 sources(hasContext=False)，再首推免责 delta，随后流式吐字，最后 done(hasAnswer=True)。"""
    with patch("app.services.chat._retrieve", new_callable=AsyncMock) as mock_retrieve, \
         patch("app.services.chat.llm_service.chat_stream") as mock_llm_stream:
        mock_retrieve.return_value = []

        async def fake_stream(*args, **kwargs):
            yield "Vue 3 采用了 Proxy 响应式系统，"
            yield "相比 Vue 2 性能大幅提升。"

        mock_llm_stream.side_effect = fake_stream

        events = []
        async for evt in chat.chat_stream(
            knowledge_base_id="kb-test",
            query="Vue 3 的优势是什么？",
        ):
            events.append(evt)

        # 校验事件序列与内容
        assert events[0]["type"] == "sources"
        assert events[0]["hasContext"] is False
        assert events[0]["sources"] == []

        assert events[1]["type"] == "delta"
        assert events[1]["text"] == _FALLBACK_PREFIX

        delta_texts = [e["text"] for e in events if e["type"] == "delta"]
        full_deltas = "".join(delta_texts)
        assert _FALLBACK_PREFIX in full_deltas
        assert "Vue 3 采用了 Proxy 响应式系统" in full_deltas

        done_event = events[-1]
        assert done_event["type"] == "done"
        assert done_event["hasAnswer"] is True
        assert done_event["answer"].startswith(_FALLBACK_PREFIX)
        assert "性能大幅提升" in done_event["answer"]


@pytest.mark.anyio
async def test_chat_with_sources_does_not_use_fallback_prefix():
    """命中了私域资料时：正常基于资料回答，不应附加降级免责前缀。"""
    with patch("app.services.chat._retrieve", new_callable=AsyncMock) as mock_retrieve, \
         patch("app.services.chat.llm_service.chat", new_callable=AsyncMock) as mock_llm_chat:
        mock_retrieve.return_value = [
            {"documentId": 1, "chunkIndex": 0, "content": "本项目采用 Spring Boot 3.3 构建。"}
        ]
        mock_llm_chat.return_value = "根据资料，项目采用 Spring Boot 3.3 构建。"

        result = await chat.chat(
            knowledge_base_id="kb-test",
            query="项目用什么框架？",
        )

        assert result["hasContext"] is True
        assert result["hasAnswer"] is True
        assert len(result["sources"]) == 1
        assert _FALLBACK_PREFIX not in result["answer"]
        assert "Spring Boot 3.3" in result["answer"]
