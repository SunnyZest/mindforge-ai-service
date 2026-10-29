"""ai_config.snapshot()：给管理端看的「当前生效配置」。

要守的四件事：
  ① 覆盖【所有】已定义的 scene —— 没配的也要出现（source=env），
     否则管理端不知道「这个用途在用默认值」；
  ② 绝不泄露 api_key，只回 has_key 布尔；
  ③ refreshedAt 只在【成功】时更新，失败保留上一次（排查「改了没生效」要看它）；
  ④ 时间带 +08:00，前端能直接显示。
"""

import json
import re

import pytest

from app.config import settings
from app.services import ai_config


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """每个用例都从「从没拉到过」开始。"""
    monkeypatch.setattr(ai_config, "_cache", {})
    monkeypatch.setattr(ai_config, "_last_error", None)
    monkeypatch.setattr(ai_config, "_refreshed_at", None)


def stub_java(monkeypatch, models=None, *, error=None):
    async def _list():
        if error is not None:
            raise error
        return models or []

    monkeypatch.setattr(ai_config.java_client, "list_ai_models", _list)


def java_model(**overrides):
    raw = {
        "id": "1",
        "scene": "CHAT",
        "provider": "dashscope",
        "modelName": "java-chat-model",
        "baseUrl": "https://java.example.invalid/v1",
        "apiKey": "sk-from-java",
    }
    raw.update(overrides)
    return raw


def refresh():
    import asyncio

    return asyncio.run(ai_config.refresh_once())


def by_scene(snapshot):
    return {item["scene"]: item for item in snapshot["scenes"]}


# ---------- ① 覆盖所有 scene ----------

def test_covers_every_defined_scene_when_nothing_configured():
    """表里一条都没有时，三个用途也必须都出现 —— 全部回退 env。"""
    snap = ai_config.snapshot()

    assert [item["scene"] for item in snap["scenes"]] == list(ai_config.ALL_SCENES)
    assert {item["source"] for item in snap["scenes"]} == {"env"}


def test_env_fallback_reports_env_model_names():
    snap = by_scene(ai_config.snapshot())

    assert snap["CHAT"]["model"] == settings.chat_model
    assert snap["EMBEDDING"]["model"] == settings.embedding_model


def test_rerank_has_no_env_fallback_and_reports_empty_model():
    """RERANK 没有环境变量兜底 —— model 空串表示「未配置」，不是「配了个空名字」。"""
    rerank = by_scene(ai_config.snapshot())["RERANK"]

    assert rerank["model"] == ""
    assert rerank["source"] == "env"
    assert rerank["hasKey"] is False


def test_java_config_wins_and_marks_source_java(monkeypatch):
    stub_java(monkeypatch, [java_model()])
    refresh()

    chat = by_scene(ai_config.snapshot())["CHAT"]

    assert chat["source"] == "java"
    assert chat["model"] == "java-chat-model"
    assert chat["baseUrl"] == "https://java.example.invalid/v1"
    assert chat["hasKey"] is True


def test_unconfigured_scene_stays_env_while_others_are_java(monkeypatch):
    stub_java(monkeypatch, [java_model(scene="CHAT")])
    refresh()

    snap = by_scene(ai_config.snapshot())
    assert snap["CHAT"]["source"] == "java"
    assert snap["EMBEDDING"]["source"] == "env"


# ---------- ② 不泄露 Key ----------

def test_snapshot_never_contains_api_key(monkeypatch):
    secret = "sk-super-secret-value-1234"
    stub_java(monkeypatch, [java_model(apiKey=secret)])
    refresh()

    dumped = json.dumps(ai_config.snapshot(), ensure_ascii=False)

    assert secret not in dumped
    assert "apiKey" not in dumped
    assert "hasKey" in dumped


def test_missing_key_reports_has_key_false(monkeypatch):
    """没填 Key 的配置（本地模型）要能下发，快照里如实说「没有 Key」。"""
    stub_java(monkeypatch, [java_model(apiKey=None)])
    refresh()

    assert by_scene(ai_config.snapshot())["CHAT"]["hasKey"] is False


# ---------- ③ refreshedAt ----------

def test_refreshed_at_is_none_before_first_success():
    assert ai_config.snapshot()["refreshedAt"] is None


def test_refreshed_at_set_after_success_with_beijing_offset(monkeypatch):
    stub_java(monkeypatch, [java_model()])
    refresh()

    value = ai_config.snapshot()["refreshedAt"]

    assert value is not None
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", value), value


def test_failure_keeps_previous_refreshed_at(monkeypatch):
    """★ 拉取失败不能把「上次成功时间」清掉 —— 那正是排查时要看的信息。"""
    stub_java(monkeypatch, [java_model()])
    refresh()
    first = ai_config.snapshot()["refreshedAt"]

    stub_java(monkeypatch, error=RuntimeError("connection refused"))
    refresh()

    snap = ai_config.snapshot()
    assert snap["refreshedAt"] == first
    assert snap["lastError"] is not None


def test_failure_before_any_success_keeps_refreshed_at_null(monkeypatch):
    stub_java(monkeypatch, error=RuntimeError("boom"))
    refresh()

    snap = ai_config.snapshot()
    assert snap["refreshedAt"] is None
    assert snap["lastError"] is not None


def test_last_error_cleared_on_next_success(monkeypatch):
    stub_java(monkeypatch, error=RuntimeError("boom"))
    refresh()
    assert ai_config.snapshot()["lastError"] is not None

    stub_java(monkeypatch, [java_model()])
    refresh()
    assert ai_config.snapshot()["lastError"] is None
