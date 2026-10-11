"""rag.search 接入 rerank 的编排层测试（mock 掉向量库 / 嵌入 / rerank HTTP）。"""

import pytest

from app.services import ai_config, ai_usage, rag
from app.services.rerank import RerankCallError


@pytest.fixture()
def clean_state(monkeypatch):
    monkeypatch.setattr(ai_config, "_cache", {})
    monkeypatch.setattr(ai_config, "_last_error", None)
    yield


def stub_rag(monkeypatch, *, vector_results, rerank_results=None, rerank_error=None):
    """把 rag 依赖全部换成可控假件。vector_results 是向量序，rerank_results 是精排后的 [(index, score)]。"""

    async def fake_embed(text, *, scene):
        return [0.1] * 4

    async def fake_query(**kwargs):
        return [dict(r) for r in vector_results]

    async def fake_rerank(query, documents, top_n=None):
        if rerank_error is not None:
            raise rerank_error
        return list(rerank_results or [])

    monkeypatch.setattr(rag.embedding_service, "embed_one", fake_embed)
    monkeypatch.setattr(rag.vector_store, "query", fake_query)
    monkeypatch.setattr(rag.rerank_service, "rerank", fake_rerank)


def stub_rerank_config(monkeypatch, model="gte-rerank-v2"):
    monkeypatch.setattr(ai_config, "_cache", {
        ai_usage.SCENE_RERANK: ai_config.ModelConfig(
            scene=ai_usage.SCENE_RERANK, api_key="sk-test",
            base_url="https://x.example.com", model=model),
    })


def _vectors(n):
    return [{"chunkId": str(i), "content": f"片段{i}", "score": round(0.9 - i * 0.1, 2)}
            for i in range(n)]


@pytest.mark.anyio
async def test_rerank_disabled_returns_vector_order(clean_state, monkeypatch):
    stub_rag(monkeypatch, vector_results=_vectors(3))

    results, reranked = await rag.search(
        knowledge_base_id="1", query="问题", top_k=2, enable_rerank=False)

    assert reranked is False
    assert [r["chunkId"] for r in results] == ["0", "1"]


@pytest.mark.anyio
async def test_rerank_without_config_degrades_cleanly(clean_state, monkeypatch):
    stub_rag(monkeypatch, vector_results=_vectors(3))
    # 没配 RERANK（_cache 为空 → env 兜底 model 为空）

    sent = []
    async def must_not_call(*args, **kwargs):
        sent.append(args)
        raise AssertionError("未配置 RERANK 时不该调 rerank")
    monkeypatch.setattr(rag.rerank_service, "rerank", must_not_call)

    results, reranked = await rag.search(
        knowledge_base_id="1", query="问题", top_k=2, enable_rerank=True)

    assert not sent
    assert reranked is False
    assert [r["chunkId"] for r in results] == ["0", "1"]


@pytest.mark.anyio
async def test_rerank_reorders_and_replaces_scores(clean_state, monkeypatch):
    stub_rag(monkeypatch,
             vector_results=_vectors(4),
             rerank_results=[(2, 0.95), (0, 0.5), (3, 0.1)])
    stub_rerank_config(monkeypatch)

    results, reranked = await rag.search(
        knowledge_base_id="1", query="问题", top_k=3, enable_rerank=True)

    assert reranked is True
    # 精排序生效，score 换成了 rerank 的相关性分数
    assert [(r["chunkId"], r["score"]) for r in results] == [
        ("2", 0.95), ("0", 0.5), ("3", 0.1)]


@pytest.mark.anyio
async def test_rerank_failure_degrades_to_vector_order(clean_state, monkeypatch):
    stub_rag(monkeypatch,
             vector_results=_vectors(4),
             rerank_error=RerankCallError("模型超时"))
    stub_rerank_config(monkeypatch)

    results, reranked = await rag.search(
        knowledge_base_id="1", query="问题", top_k=2, enable_rerank=True)

    assert reranked is False
    # 降级回向量序并截到 topK
    assert [r["chunkId"] for r in results] == ["0", "1"]
