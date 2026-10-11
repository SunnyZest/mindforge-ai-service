"""rerank 模块的单元测试（fake HTTP，不打真实模型）。"""

import httpx
import pytest

from app.services import ai_config, ai_usage, rerank


@pytest.fixture()
def clean_state(monkeypatch):
    """把 ai_config 缓存清空，让 RERANK 回到「未配置」的基线。"""
    monkeypatch.setattr(ai_config, "_cache", {})
    monkeypatch.setattr(ai_config, "_last_error", None)
    yield


def stub_config(monkeypatch, *, model="gte-rerank-v2", api_key="sk-test",
                base_url="https://dashscope.example.com/compatible-mode/v1"):
    monkeypatch.setattr(ai_config, "_cache", {
        ai_usage.SCENE_RERANK: ai_config.ModelConfig(
            scene=ai_usage.SCENE_RERANK, api_key=api_key,
            base_url=base_url, model=model, provider="dashscope"),
    })


class StubTransport(httpx.AsyncBaseTransport):
    """按注册的响应逐次返回，并记录收到的请求。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request):
        self.requests.append(request)
        entry = self.responses.pop(0)
        if isinstance(entry, Exception):
            raise entry
        status, body = entry
        return httpx.Response(status, json=body, request=request)


@pytest.mark.anyio
async def test_unconfigured_returns_empty_and_skips_http(monkeypatch, clean_state):
    sent = []

    class Guard(StubTransport):
        async def handle_async_request(self, request):
            sent.append(request)
            raise AssertionError("未配置时不该发请求")

    monkeypatch.setattr(rerank.httpx, "AsyncClient",
                        lambda **kw: _client_with(Guard([])))

    assert rerank.configured() is False
    with pytest.raises(rerank.RerankCallError):
        await rerank.rerank("问题", ["候选一"])
    assert not sent


@pytest.mark.anyio
async def test_rerank_parses_results_sorted_by_score(monkeypatch, clean_state):
    transport = StubTransport([
        (200, {"output": {"results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.4},
        ]}, "usage": {"total_tokens": 12}}),
    ])
    monkeypatch.setattr(rerank.httpx, "AsyncClient",
                        lambda **kw: _client_with(transport))
    stub_config(monkeypatch)

    ranked = await rerank.rerank("问题", ["候选一", "候选二", "候选三"], top_n=2)

    # 按分数降序
    assert [index for index, _ in ranked] == [2, 0]
    # 端点剥掉了兼容层路径，拼上原生 rerank 路径
    path = str(transport.requests[0].url)
    assert path.endswith("/api/v1/services/rerank/text-rerank/text-rerank")
    assert "/compatible-mode" not in path
    # 报文结构（camelCase 不适用 —— DashScope 原生 API 用它自己的字段名）
    body = transport.requests[0].read().decode()
    assert '"top_n":2' in body and '"return_documents":false' in body


@pytest.mark.anyio
async def test_http_error_raises_rerank_call_error(monkeypatch, clean_state):
    transport = StubTransport([(500, {"message": "boom"})])
    monkeypatch.setattr(rerank.httpx, "AsyncClient",
                        lambda **kw: _client_with(transport))
    stub_config(monkeypatch)

    with pytest.raises(rerank.RerankCallError):
        await rerank.rerank("问题", ["候选一"])


@pytest.mark.anyio
async def test_malformed_body_raises(monkeypatch, clean_state):
    transport = StubTransport([(200, {"output": {}})])
    monkeypatch.setattr(rerank.httpx, "AsyncClient",
                        lambda **kw: _client_with(transport))
    stub_config(monkeypatch)

    with pytest.raises(rerank.RerankCallError):
        await rerank.rerank("问题", ["候选一"])


@pytest.mark.anyio
async def test_empty_documents_short_circuits(monkeypatch, clean_state):
    stub_config(monkeypatch)
    assert await rerank.rerank("问题", []) == []


class _ClientStub:
    """最小化的 AsyncClient 替身：只实现 post + 异步上下文协议。"""

    def __init__(self, transport):
        self._transport = transport

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, **kwargs):
        request = httpx.Request("POST", url, **{
            "json": kwargs.get("json"),
            "headers": kwargs.get("headers"),
        })
        return await self._transport.handle_async_request(request)


def _client_with(transport):
    return _ClientStub(transport)
