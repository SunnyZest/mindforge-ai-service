"""ai_config：Java 下发的模型配置 + 环境变量兜底。

不需要起服务 —— 把 java_client.list_ai_models 换成假的就行。
要守的两件事：
  ① 「DB 是覆盖层，环境变量是默认值」—— 表空 / 拉不到 / 配置删了，都要能回退；
  ② 拉取失败【不能清空缓存】—— 否则 Java 抖一下所有模型就一起换了。
"""

import asyncio

import pytest

from app.config import settings
from app.services import ai_config

SCENE = ai_config.SCENE_CHAT


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """每个用例都从「从没拉到过」开始，跑完清干净，免得互相污染。"""
    monkeypatch.setattr(ai_config, "_cache", {})
    monkeypatch.setattr(ai_config, "_last_error", None)
    yield
    ai_config._cache = {}
    ai_config._last_error = None


def stub_java(monkeypatch, models=None, *, error=None):
    """把 java_client.list_ai_models 换成假的。"""

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
        "temperature": 0.2,
        "maxTokens": 4096,
        "timeoutSeconds": 300,
        "isDefault": True,
    }
    raw.update(overrides)
    return raw


def refresh():
    return asyncio.run(ai_config.refresh_once())


# ---------- ① 兜底 ----------

def test_falls_back_to_env_when_nothing_cached():
    config = ai_config.get(SCENE)

    assert config.source == "env"
    assert config.model == settings.chat_model
    assert config.api_key == settings.dashscope_api_key
    assert config.base_url == settings.dashscope_base_url


def test_embedding_falls_back_to_embedding_model():
    config = ai_config.get(ai_config.SCENE_EMBEDDING)

    assert config.source == "env"
    assert config.model == settings.embedding_model


def test_get_never_returns_none():
    """调用方不该写 `if config is None` —— 任何 scene 都要有兜底。"""
    for scene in (SCENE, ai_config.SCENE_EMBEDDING, ai_config.SCENE_RERANK, "UNKNOWN_SCENE"):
        assert ai_config.get(scene) is not None


# ---------- ② Java 下发生效 ----------

def test_java_config_overrides_env(monkeypatch):
    stub_java(monkeypatch, [java_model()])

    assert refresh() == 1

    config = ai_config.get(SCENE)
    assert config.source == "java"
    assert config.model == "java-chat-model"
    assert config.api_key == "sk-from-java"
    assert config.base_url == "https://java.example.invalid/v1"
    assert config.temperature == 0.2
    assert config.max_tokens == 4096


def test_only_configured_scenes_are_overridden(monkeypatch):
    """只配了 CHAT 时，EMBEDDING 要继续用环境变量。"""
    stub_java(monkeypatch, [java_model(scene="CHAT")])
    refresh()

    assert ai_config.get(SCENE).source == "java"
    assert ai_config.get(ai_config.SCENE_EMBEDDING).source == "env"


def test_duplicate_scene_keeps_first(monkeypatch):
    """Java 侧按「默认优先」排好序了，所以第一条胜出。"""
    stub_java(monkeypatch, [
        java_model(modelName="默认的", isDefault=True),
        java_model(id="2", modelName="备用的", isDefault=False),
    ])
    refresh()

    assert ai_config.get(SCENE).model == "默认的"


def test_deleting_config_falls_back_to_env(monkeypatch):
    """配置被删掉 → 缓存整体替换 → 自动回退环境变量（不留残影）。"""
    stub_java(monkeypatch, [java_model()])
    refresh()
    assert ai_config.get(SCENE).source == "java"

    stub_java(monkeypatch, [])
    refresh()
    assert ai_config.get(SCENE).source == "env"
    assert ai_config.get(SCENE).model == settings.chat_model


# ---------- ③ 容错 ----------

def test_pull_failure_keeps_previous_cache(monkeypatch):
    """★ Java 抖一下不能让所有模型一起换掉。"""
    stub_java(monkeypatch, [java_model()])
    refresh()

    stub_java(monkeypatch, error=RuntimeError("connection refused"))
    assert refresh() == 0

    config = ai_config.get(SCENE)
    assert config.source == "java", "拉取失败时必须沿用上一次的配置"
    assert config.model == "java-chat-model"
    assert ai_config.last_error() is not None


def test_pull_failure_before_any_success_falls_back_to_env(monkeypatch):
    stub_java(monkeypatch, error=RuntimeError("connection refused"))

    assert refresh() == 0
    assert ai_config.get(SCENE).source == "env"


def test_last_error_is_cleared_on_success(monkeypatch):
    stub_java(monkeypatch, error=RuntimeError("boom"))
    refresh()
    assert ai_config.last_error() is not None

    stub_java(monkeypatch, [java_model()])
    refresh()
    assert ai_config.last_error() is None


# ---------- ④ 脏数据 ----------

@pytest.mark.parametrize("bad", [
    {"scene": "", "modelName": "x"},
    {"scene": "CHAT", "modelName": ""},
    {"scene": None, "modelName": None},
])
def test_incomplete_config_is_skipped(monkeypatch, bad):
    stub_java(monkeypatch, [java_model(**bad), java_model(id="2", modelName="好的")])
    refresh()

    assert ai_config.get(SCENE).model == "好的", "不完整的配置要跳过，不能让整批都作废"


def test_blank_base_url_uses_default(monkeypatch):
    stub_java(monkeypatch, [java_model(baseUrl=None)])
    refresh()

    assert ai_config.get(SCENE).base_url == settings.dashscope_base_url


def test_null_api_key_becomes_empty_string(monkeypatch):
    """没填 Key 的配置（比如本地模型）要能下发，apiKey 归一成空串。"""
    stub_java(monkeypatch, [java_model(apiKey=None)])
    refresh()

    assert ai_config.get(SCENE).api_key == ""


def test_scene_is_normalized_to_upper(monkeypatch):
    stub_java(monkeypatch, [java_model(scene=" chat ")])
    refresh()

    assert ai_config.get(SCENE).source == "java"


# ---------- ⑤ 别把 Key 打出去 ----------

def test_describe_never_contains_api_key(monkeypatch):
    stub_java(monkeypatch, [java_model(apiKey="sk-super-secret-value")])
    refresh()

    described = str(ai_config.describe())
    assert "sk-super-secret-value" not in described
    assert "java-chat-model" in described
    assert "java" in described
