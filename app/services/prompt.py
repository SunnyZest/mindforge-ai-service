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


def render(name: str, **kwargs) -> str:
    """读取 prompt 并把占位符替换成实际内容。

    Args:
        name: 文件名（不含 .txt）
        **kwargs: 占位符的值，如 context="..."

    Raises:
        PromptNotFoundError: 文件不存在
        PromptRenderError: 有占位符没填，或者填了多余的变量

    ⚠️ 这里刻意【不用 str.format()】，而是逐个替换已知变量。原因：
        prompt 里经常要写 JSON 示例（比如简历解析要模型输出
        {"name": "...", "skills": [...]}），用 format() 的话
        那些花括号会被当成占位符，直接抛 KeyError。
        逐个替换只动我们认识的变量，其余花括号原样保留。

    ⚠️ 也刻意【不做缓存】—— 读个小文件不到 1ms，而一次对话本来就要几秒。
       不缓存的代价可以忽略，换来的是「改完 prompt 立刻生效」，
       开发时不用重启服务。
    """
    text = load(name)

    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", "" if value is None else str(value))

    # 替换完还剩下的占位符 = 调用方漏传了。
    # 必须报错 —— 否则模型会收到一个字面量 {context}，然后基于它胡说，
    # 而你要盯着回答想半天「为什么它答得不对劲」。
    missing = _PLACEHOLDER.findall(text)
    if missing:
        raise PromptRenderError(
            f"Prompt「{name}」有占位符没填：{sorted(set(missing))}；"
            f"本次传入的是：{sorted(kwargs)}"
        )

    # 传了多余的变量：说明调用方和 prompt 已经对不上了，提醒一下
    unused = set(kwargs) - set(_PLACEHOLDER.findall(load(name)))
    if unused:
        logger.warning("Prompt「%s」收到未使用的变量：%s", name, sorted(unused))

    return text
