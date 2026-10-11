"""Prompt 加载与渲染。

所有提示词以 .txt 文件放在项目根目录的 prompts/ 下，按名字取用。
这样改文案不用动 Python 代码，git diff 也能一眼看出改的是哪句话。

⚠️ 这个模块是 Prompt 的【唯一入口】。
将来阶段 18 要做「管理端改 prompt + 版本管理」时，只改这里
（改成从数据库读），调用方一行都不用动。
"""

import logging
import re
from pathlib import Path

from app.config import PROJECT_ROOT
from app.services import prompt_store

logger = logging.getLogger(__name__)

PROMPTS_DIR = PROJECT_ROOT / "prompts"

# 占位符形如 {context} / {question}
_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class PromptNotFoundError(Exception):
    """Prompt 文件不存在。"""


class PromptRenderError(Exception):
    """变量没填对。"""


def load(name: str) -> str:
    """按名字读取 prompt 原文（不渲染）。

    Args:
        name: 文件名（不含 .txt），如 "rag_chat"

    Raises:
        PromptNotFoundError: 文件不存在
    """
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.is_file():
        raise PromptNotFoundError(
            f"Prompt 不存在：{path}。"
            f"可用的是：{sorted(p.stem for p in PROMPTS_DIR.glob('*.txt'))}"
        )
    return path.read_text(encoding="utf-8")


def render(name: str, *, prompt_id: str | None = None, **kwargs) -> str:
    """读取 prompt 并把占位符替换成实际内容。

    Args:
        name: 场景名（不含 .txt），如 "rag_chat"
        prompt_id: 指定用哪一条（管理端配置的版本行 id）。
            不传就用该场景的默认那条。传了但找不到 / 没开放给用户切换时，
            prompt_store 会回退默认并打 WARNING。
            ⚠️ 这是**保留参数名**，不会被当成占位符 —— prompt 里别写 {prompt_id}。
        **kwargs: 占位符的值，如 context="..."

    Raises:
        PromptNotFoundError: 管理端没配、文件也不存在
        PromptRenderError: 有占位符没填，或者填了多余的变量

    ⚠️ 这里刻意【不用 str.format()】，而是逐个替换已知变量。原因：
        prompt 里经常要写 JSON 示例（比如简历解析要模型输出
        {"name": "...", "skills": [...]}），用 format() 的话
        那些花括号会被当成占位符，直接抛 KeyError。
        逐个替换只动我们认识的变量，其余花括号原样保留。

    ⚠️ 也刻意【不做缓存】—— 缓存由 prompt_store 负责（它拉管理端配置）。
       这里只做「取模板 + 替换」。
    """
    template, source = _template(name, prompt_id)

    text = template
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", "" if value is None else str(value))

    # 替换完还剩下的占位符 = 调用方漏传了（或者管理端配的那版占位符和代码对不上）。
    # 必须报错 —— 否则模型会收到一个字面量 {context}，然后基于它胡说，
    # 而你要盯着回答想半天「为什么它答得不对劲」。
    missing = _PLACEHOLDER.findall(text)
    if missing:
        raise PromptRenderError(
            f"Prompt「{name}」（{source}）有占位符没填：{sorted(set(missing))}；"
            f"本次传入的是：{sorted(kwargs)}"
        )

    # 传了多余的变量：说明调用方和 prompt 已经对不上了，提醒一下
    unused = set(kwargs) - set(_PLACEHOLDER.findall(template))
    if unused:
        logger.warning("Prompt「%s」（%s）收到未使用的变量：%s", name, source, sorted(unused))

    return text


def _template(name: str, prompt_id: str | None) -> tuple[str, str]:
    """取模板正文，以及它的来源（来源会带进日志和报错，方便定位是哪一版出的问题）。

    ⚠️ 管理端配的那条如果占位符和调用方对不上，**不自动回退文件版** ——
       静默回退会让管理员以为「改了、生效了」，实际跑的还是旧文件。
       报错（由上面的 PromptRenderError 抛出）才能让他知道配错了。
    """
    entry = prompt_store.find(name, prompt_id)
    if entry is not None:
        return entry.content, f"管理端配置「{entry.name}」v{entry.version}"
    return load(name), f"文件 prompts/{name}.txt"
