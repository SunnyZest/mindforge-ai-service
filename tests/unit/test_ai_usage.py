"""ai_usage：AI 使用量的记账。

这个模块的可靠性约定是「尽力而为」—— 记账失败绝不能影响业务。
所以用例重点不在「记得准不准」，而在「它出问题的时候会不会把调用方拖下水」。
"""

import asyncio
from types import SimpleNamespace

import pytest

from app.services import ai_usage


@pytest.fixture(autouse=True)
def clean_context():
    """contextvar 是全局的，每个用例都从空开始。"""
    ai_usage.set_trace_id("")
    ai_usage.set_user_id("")
    yield
    ai_usage.set_trace_id("")
    ai_usage.set_user_id("")


def capture(monkeypatch):
    """把上报换成「记下来」，并返回那个列表。"""
    sent: list[dict] = []

    async def fake_report(payload):
        sent.append(payload)

    monkeypatch.setattr(ai_usage.java_client, "report_ai_usage", fake_report)
    return sent


def usage(prompt=10, completion=5, total=15):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


# ---------- usage_fields ----------

def test_usage_fields_from_normal_response():
    assert ai_usage.usage_fields(usage(10, 5, 15)) == (10, 5, 15)


def test_usage_fields_handles_missing_completion():
    """embedding 的响应只有 prompt_tokens / total_tokens，没有 completion。"""
    assert ai_usage.usage_fields(SimpleNamespace(prompt_tokens=2, total_tokens=2)) == (2, 0, 2)


def test_usage_fields_handles_none():
    assert ai_usage.usage_fields(None) == (0, 0, 0)


def test_usage_fields_handles_null_values():
    """SDK 的字段可能是 null（不是 0）—— 直接相加会 TypeError。"""
    assert ai_usage.usage_fields(SimpleNamespace(prompt_tokens=None,
                                                completion_tokens=None,
                                                total_tokens=None)) == (0, 0, 0)


def test_usage_fields_reads_dict():
    """★ rerank 走 DashScope 原生 REST，usage 是 dict 不是对象。

    以前只用 getattr，dict 一律取到 0 —— rerank 的 token 记账永远是 0，
    而且不报错，看着像「模型没消耗」，实际是统计瞎了。
    """
    assert ai_usage.usage_fields({"prompt_tokens": 30, "completion_tokens": 0,
                                  "total_tokens": 30}) == (30, 0, 30)


def test_usage_fields_reads_dict_with_only_total():
    """DashScope rerank 只回 total_tokens，没有 prompt/completion。"""
    assert ai_usage.usage_fields({"total_tokens": 42}) == (0, 0, 42)


def test_usage_fields_tolerates_bad_dict_values():
    """脏值（字符串 / 非法数字）不能把记账整条打挂。"""
    assert ai_usage.usage_fields({"prompt_tokens": "abc", "total_tokens": None}) == (0, 0, 0)


# ---------- payload 形状 ----------

def test_payload_is_camel_case(monkeypatch):
    sent = capture(monkeypatch)

    async def run():
        ai_usage.record(scene="RAG_CHAT", provider="dashscope", model="qwen",
                        usage=usage(), duration_ms=123)
        await ai_usage.drain()

    asyncio.run(run())

    assert len(sent) == 1
    payload = sent[0]
    assert payload["scene"] == "RAG_CHAT"
    assert payload["provider"] == "dashscope"
    assert payload["modelName"] == "qwen"
    assert payload["promptTokens"] == 10
    assert payload["completionTokens"] == 5
    assert payload["totalTokens"] == 15
    assert payload["durationMs"] == 123
    assert payload["status"] == ai_usage.SUCCESS
    # Java 侧是 record，字段名对不上会静默变 null
    assert not [k for k in payload if "_" in k], f"出现 snake_case 的键：{list(payload)}"


def test_payload_carries_trace_and_user_from_context(monkeypatch):
    """traceId / userId 是中间件从 X-Trace-Id / X-User-Id 头里接住的。"""
    sent = capture(monkeypatch)

    async def run():
        ai_usage.set_trace_id("trace-abc")
        ai_usage.set_user_id("42")
        ai_usage.record(scene="RAG_CHAT", usage=usage())
        await ai_usage.drain()

    asyncio.run(run())

    assert sent[0]["requestId"] == "trace-abc"
    assert sent[0]["userId"] == "42"


def test_explicit_user_id_wins_over_context(monkeypatch):
    sent = capture(monkeypatch)

    async def run():
        ai_usage.set_user_id("42")
        ai_usage.record(scene="RAG_CHAT", user_id="99", usage=usage())
        await ai_usage.drain()

    asyncio.run(run())

    assert sent[0]["userId"] == "99"


def test_missing_context_becomes_null(monkeypatch):
    """系统自动调用（比如文档向量化）没有用户，要传 null 而不是空串。"""
    sent = capture(monkeypatch)

    async def run():
        ai_usage.record(scene="EMBEDDING", usage=usage())
        await ai_usage.drain()

    asyncio.run(run())

    assert sent[0]["userId"] is None
    assert sent[0]["requestId"] is None


def test_error_message_is_truncated(monkeypatch):
    """Java 那列是 varchar(1000)，超了整条记录会被丢掉。"""
    sent = capture(monkeypatch)

    async def run():
        ai_usage.record(scene="RAG_CHAT", status=ai_usage.FAILED, error_message="E" * 5000)
        await ai_usage.drain()

    asyncio.run(run())

    assert len(sent[0]["errorMessage"]) == 1000


# ---------- 「尽力而为」的三条约定 ----------

def test_report_failure_does_not_raise(monkeypatch):
    """★ 上报失败绝不能影响调用方 —— 一次成功的 AI 调用不能因为记不了账变成失败。"""

    async def boom(payload):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(ai_usage.java_client, "report_ai_usage", boom)

    async def run():
        ai_usage.record(scene="RAG_CHAT", usage=usage())
        await ai_usage.drain()

    asyncio.run(run())          # 不抛异常就算过


def test_record_outside_event_loop_is_noop():
    """同步上下文里调（脚本 / 单测）不能炸 —— 静默跳过就好。"""
    ai_usage.record(scene="RAG_CHAT", usage=usage())        # 不抛异常就算过


def test_record_does_not_block(monkeypatch):
    """★ 不 await：record 必须立刻返回，不能等 HTTP 往返。"""
    import time

    async def slow(payload):
        await asyncio.sleep(5)

    monkeypatch.setattr(ai_usage.java_client, "report_ai_usage", slow)

    async def run():
        started = time.monotonic()
        ai_usage.record(scene="RAG_CHAT", usage=usage())
        elapsed = time.monotonic() - started
        assert elapsed < 0.5, f"record 阻塞了 {elapsed:.2f} 秒 —— 它必须是 fire-and-forget"
        # 别等那 5 秒，直接取消
        for task in list(ai_usage._pending):
            task.cancel()

    asyncio.run(run())


def test_drain_waits_for_pending_reports(monkeypatch):
    """关服务时要等最后几条账发完，否则会丢。"""
    sent = capture(monkeypatch)

    async def run():
        for _ in range(3):
            ai_usage.record(scene="RAG_CHAT", usage=usage())
        await ai_usage.drain()

    asyncio.run(run())

    assert len(sent) == 3
    assert not ai_usage._pending, "drain 之后不该还有在飞的 task"


def test_drain_with_nothing_pending_returns_immediately():
    asyncio.run(ai_usage.drain())


def test_pending_tasks_are_referenced(monkeypatch):
    """★ 必须持有 task 的强引用 —— asyncio 对 task 只持弱引用，不持有可能被 GC 掉。"""
    capture(monkeypatch)

    async def run():
        ai_usage.record(scene="RAG_CHAT", usage=usage())
        assert ai_usage._pending, "record 之后 _pending 里应该有这个 task"
        await ai_usage.drain()

    asyncio.run(run())
