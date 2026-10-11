"""prompt_store：Java 下发的提示词 + 文件兜底。

这里最要紧的一条是**安全边界**：Java 的 /internal/prompts 会把【所有已发布】的
提示词都下发下来（Python 得能按场景取默认那条来渲染，包括不让用户切的那 7 个
内部契约 prompt）。所以「用户能不能切到某一条」必须由 prompt_store 自己再挡一道 ——
否则用户可以传一个内部 prompt 的 id，把「简历解析必须吐特定 JSON」那套换掉。
"""

import asyncio

import pytest

from app.services import prompt_store


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.setattr(prompt_store, "_by_id", {})
    monkeypatch.setattr(prompt_store, "_by_scene", {})
    monkeypatch.setattr(prompt_store, "_last_error", None)
    yield
    prompt_store._by_id = {}
    prompt_store._by_scene = {}
    prompt_store._last_error = None


def stub_java(monkeypatch, prompts=None, *, error=None):
    async def _list():
        if error is not None:
            raise error
        return prompts or []

    monkeypatch.setattr(prompt_store.java_client, "list_prompts", _list)


def entry(prompt_id="1", scene="RAG_CHAT", name="默认", version=1, *,
          selectable=True, is_default=False, content="模板 {context}"):
    return {
        "id": prompt_id,
        "scene": scene,
        "name": name,
        "version": version,
        "label": name,
        "isDefault": is_default,
        "userSelectable": selectable,
        "content": content,
    }


def refresh():
    return asyncio.run(prompt_store.refresh_once())


# ---------- 兜底 ----------

def test_find_returns_none_when_nothing_cached():
    assert prompt_store.find("RAG_CHAT") is None


def test_find_returns_none_for_unknown_scene(monkeypatch):
    stub_java(monkeypatch, [entry()])
    refresh()

    assert prompt_store.find("RESUME_PARSE") is None


# ---------- 默认选择 ----------

def test_find_prefers_default(monkeypatch):
    stub_java(monkeypatch, [
        entry("1", name="严谨", is_default=False),
        entry("2", name="默认", is_default=True),
    ])
    refresh()

    assert prompt_store.find("RAG_CHAT").name == "默认"


def test_find_falls_back_to_first_when_no_default(monkeypatch):
    stub_java(monkeypatch, [entry("1", name="甲"), entry("2", name="乙")])
    refresh()

    assert prompt_store.find("RAG_CHAT").name == "甲"


def test_scene_lookup_is_case_insensitive(monkeypatch):
    stub_java(monkeypatch, [entry(scene="RAG_CHAT")])
    refresh()

    assert prompt_store.find("rag_chat") is not None


# ---------- 指定 id ----------

def test_find_by_id(monkeypatch):
    stub_java(monkeypatch, [entry("1", name="默认", is_default=True),
                            entry("2", name="严谨")])
    refresh()

    assert prompt_store.find("RAG_CHAT", "2").name == "严谨"


def test_unknown_id_falls_back_to_default(monkeypatch):
    """用户上次选的风格被管理员下线了 —— 不该让他的提问失败。"""
    stub_java(monkeypatch, [entry("1", name="默认", is_default=True)])
    refresh()

    assert prompt_store.find("RAG_CHAT", "999").name == "默认"


def test_id_from_other_scene_falls_back(monkeypatch):
    stub_java(monkeypatch, [entry("1", name="默认", is_default=True),
                            entry("2", scene="RESUME_PARSE", name="解析")])
    refresh()

    assert prompt_store.find("RAG_CHAT", "2").name == "默认"


def test_not_selectable_id_is_rejected(monkeypatch):
    """★ 核心安全边界：用户不能切到管理员没开放给他的提示词。"""
    stub_java(monkeypatch, [
        entry("1", name="默认", is_default=True, selectable=True),
        entry("2", name="内部解析用", selectable=False),
    ])
    refresh()

    found = prompt_store.find("RAG_CHAT", "2")
    assert found is not None
    assert found.name == "默认", "传了不可选的 id 必须回退默认，而不是照用"
    assert found.id == "1"


def test_default_is_usable_even_if_not_selectable(monkeypatch):
    """默认那条是「系统默认风格」，不要求它也标 user_selectable。"""
    stub_java(monkeypatch, [entry("1", name="默认", is_default=True, selectable=False)])
    refresh()

    assert prompt_store.find("RAG_CHAT").name == "默认"


# ---------- 脏数据与容错 ----------

@pytest.mark.parametrize("bad", [
    {"scene": "", "content": "x"},
    {"scene": "RAG_CHAT", "content": ""},
    {"scene": "RAG_CHAT", "content": None},
])
def test_incomplete_entry_is_skipped(monkeypatch, bad):
    raw = entry()
    raw.update(bad)
    stub_java(monkeypatch, [raw, entry("2", name="好的")])
    refresh()

    assert prompt_store.find("RAG_CHAT").name == "好的"


def test_pull_failure_keeps_previous_cache(monkeypatch):
    """★ Java 抖一下不能让所有提示词一起掉回文件版。"""
    stub_java(monkeypatch, [entry("1", name="默认", is_default=True)])
    refresh()

    stub_java(monkeypatch, error=RuntimeError("connection refused"))
    assert refresh() == 0

    assert prompt_store.find("RAG_CHAT").name == "默认"
    assert prompt_store.last_error() is not None


def test_deleting_all_prompts_clears_cache(monkeypatch):
    stub_java(monkeypatch, [entry()])
    refresh()
    assert prompt_store.find("RAG_CHAT") is not None

    stub_java(monkeypatch, [])
    refresh()
    assert prompt_store.find("RAG_CHAT") is None, "配置删了要能回退文件版（不留残影）"


def test_last_error_cleared_on_success(monkeypatch):
    stub_java(monkeypatch, error=RuntimeError("boom"))
    refresh()
    assert prompt_store.last_error() is not None

    stub_java(monkeypatch, [entry()])
    refresh()
    assert prompt_store.last_error() is None


# ---------- 日志安全 ----------

def test_describe_has_no_content(monkeypatch):
    stub_java(monkeypatch, [entry(content="这是不该出现在日志里的提示词正文 {context}")])
    refresh()

    described = str(prompt_store.describe())
    assert "不该出现在日志里" not in described
    assert "默认" in described
