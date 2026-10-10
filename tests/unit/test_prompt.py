"""prompt 加载与渲染。

其中两个用例是「自维护」的：它们遍历 prompts/ 下所有 .txt，
所以以后新加一个 prompt 会自动被覆盖，不用记得来这里补一行。
"""

import re

import pytest

from app.services import prompt

# 和 prompt.py 里用的是同一个规则
PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def all_prompt_names() -> list[str]:
    return sorted(p.stem for p in prompt.PROMPTS_DIR.glob("*.txt"))


def test_prompts_dir_exists():
    assert prompt.PROMPTS_DIR.is_dir(), f"prompts 目录不存在：{prompt.PROMPTS_DIR}"
    assert all_prompt_names(), "prompts/ 下一个 .txt 都没有"


@pytest.mark.parametrize("name", all_prompt_names())
def test_every_prompt_renders_cleanly(name: str):
    """★ 自维护用例：每个 prompt 用它自己声明的占位符渲染，渲染后不该剩任何 {xxx}。

    这条能挡住两类问题：① 提示词里手滑写了 {resumeText} 这种没被替换的占位符，
    模型会收到字面量然后基于它胡说；② 新加的 prompt 里写了不该写的花括号。
    """
    text = prompt.load(name)
    kwargs = {key: f"<{key}>" for key in set(PLACEHOLDER.findall(text))}

    rendered = prompt.render(name, **kwargs)

    leftovers = PLACEHOLDER.findall(rendered)
    assert not leftovers, f"{name} 渲染后还有未替换的占位符：{sorted(set(leftovers))}"


@pytest.mark.parametrize("name", all_prompt_names())
def test_json_examples_survive_rendering(name: str):
    """★ 提示词里的 JSON 示例（{"name": ...}）不能被当成占位符吃掉。

    这是 render 刻意不用 str.format() 的原因 —— 用了的话那些花括号会被当成
    占位符、直接抛 KeyError。改成 format 的人会在这里被挡住。
    """
    text = prompt.load(name)
    if "{" not in text or '"' not in text:
        pytest.skip("这个 prompt 里没有 JSON 示例")

    kwargs = {key: "v" for key in set(PLACEHOLDER.findall(text))}
    rendered = prompt.render(name, **kwargs)

    assert re.search(r'"\w+"\s*:', rendered), (
        f"{name} 渲染后 JSON 示例被破坏了 —— 检查 render 是不是被改成了 str.format()"
    )


def test_missing_placeholder_raises():
    """漏传占位符必须报错，不能留一个字面量 {xxx} 给模型。"""
    name = "resume_parse"
    keys = sorted(set(PLACEHOLDER.findall(prompt.load(name))))
    assert len(keys) >= 2, f"{name} 应该至少有两个占位符，实际 {keys}"

    partial = {key: "v" for key in keys[:-1]}      # 故意少传一个
    with pytest.raises(prompt.PromptRenderError) as excinfo:
        prompt.render(name, **partial)

    message = str(excinfo.value)
    assert keys[-1] in message, "报错信息里要指出是哪个占位符没填"
    assert name in message, "报错信息里要指出是哪个 prompt"


def test_unknown_prompt_raises_with_available_list():
    with pytest.raises(prompt.PromptNotFoundError) as excinfo:
        prompt.load("definitely_not_a_prompt")

    # 报错信息里要列出可用的名字，否则还得去翻目录
    message = str(excinfo.value)
    assert "resume_parse" in message or "rag_chat" in message


def test_none_value_renders_as_empty_string():
    """显式传 None 时渲染成空串（而不是 "None" 四个字符）。"""
    name = "resume_parse"
    keys = sorted(set(PLACEHOLDER.findall(prompt.load(name))))
    kwargs = {key: None for key in keys}

    rendered = prompt.render(name, **kwargs)

    assert "None" not in rendered
    assert not PLACEHOLDER.findall(rendered)
