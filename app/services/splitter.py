"""文本分块。

为什么要分块：LLM 的上下文有限，而且检索时按「片段」召回比按「整篇」精准得多。

参数（800 / 150）来自 API 文档 §十七 的建议值。
原 agent 项目用的是 200 / 20 —— 太碎了，检索回来的片段上下文不够，
LLM 容易答不准，所以这里没有沿用。
"""

import logging
import re

logger = logging.getLogger(__name__)

# 优先在这些位置切分：段落 → 换行 → 句号 → 分号 → 逗号
# 顺序即优先级，越靠前越是「自然的边界」
_SPLIT_PATTERNS = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "; ", "，", ", "]


def split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """把长文本切成带重叠的片段。

    策略：
      1. 先按「自然边界」把文本拆成原子片段（尽量不切断句子）
      2. 再把这些片段按 chunk_size 聚合
      3. 相邻片段之间保留 chunk_overlap 个字符的重叠

    重叠的作用：避免关键信息刚好落在切分点上、两边都不完整。

    Args:
        text: 纯文本
        chunk_size: 单个片段的目标字符数
        chunk_overlap: 相邻片段的重叠字符数

    Returns:
        片段列表；文本本身很短时返回单元素列表
    """
    text = text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})")

    # 文本本身就够短，不用切
    if len(text) <= chunk_size:
        return [text]

    atoms = _split_to_atoms(text)

    chunks: list[str] = []
    current = ""

    for atom in atoms:
        # 单个原子就超过 chunk_size（比如一整段没有标点的长文）→ 硬切
        if len(atom) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(atom, chunk_size, chunk_overlap))
            continue

        if len(current) + len(atom) <= chunk_size:
            current += atom
        else:
            if current:
                chunks.append(current)
            # 用上一段的尾部作为重叠，保证跨片段的上下文连续
            tail = current[-chunk_overlap:] if chunk_overlap and current else ""
            current = tail + atom

    if current:
        chunks.append(current)

    # 清理：去掉纯空白片段
    chunks = [c.strip() for c in chunks if c.strip()]

    logger.debug("分块完成：原文 %d 字符 → %d 个片段", len(text), len(chunks))
    return chunks


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。

    不追求精确 —— 这个值只用于统计展示，不影响检索。
    经验值：中文 1 字 ≈ 1 token，英文 4 字符 ≈ 1 token。
    """
    if not text:
        return 0
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - chinese
    # 注意 max(1, ...) 要包住【整体】，不是只包住 other // 4 ——
    # 后者会让纯中文文本凭空多出 1 个 token。
    return max(1, chinese + other // 4)


# ---------- 内部 ----------


def _split_to_atoms(text: str) -> list[str]:
    """按自然边界把文本拆成原子片段。

    用递归下降：先用最优先的分隔符切，切出来的块如果还很大，
    再用下一个分隔符继续切。
    """
    return _recursive_split(text, 0)


def _recursive_split(text: str, level: int) -> list[str]:
    if level >= len(_SPLIT_PATTERNS):
        return [text]

    sep = _SPLIT_PATTERNS[level]
    parts = text.split(sep)

    if len(parts) == 1:
        # 这个分隔符切不开，换下一个
        return _recursive_split(text, level + 1)

    result: list[str] = []
    for i, part in enumerate(parts):
        if not part:
            continue
        # 把分隔符补回去（除了最后一段），保持原文完整
        piece = part + sep if i < len(parts) - 1 else part
        result.append(piece)
    return result


def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """没有任何自然边界可用时的兜底：按固定长度硬切。"""
    step = max(1, chunk_size - chunk_overlap)
    return [text[i:i + chunk_size] for i in range(0, len(text), step) if text[i:i + chunk_size].strip()]
