"""文档解析：把文件字节变成纯文本。

目前支持纯文本类（TXT / MD）。
PDF / DOCX / HTML 尚未支持 —— PDF 解析的坑最多（扫描件要 OCR、双栏排版会乱、
表格会丢），需要单独处理，所以先不纳入。
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 支持的扩展名（小写）。Java 侧允许 PDF/DOCX/TXT/MD/HTML，
# 但本服务只处理纯文本类，其余会明确报错 —— 报错好过静默产出垃圾数据。
SUPPORTED_TEXT_TYPES = {"TXT", "MD", "MARKDOWN"}


class UnsupportedFileTypeError(Exception):
    """文件类型暂不支持。"""


@dataclass
class ParseResult:
    text: str
    page_count: int | None = None


def parse(content: bytes, file_type: str) -> ParseResult:
    """把文件内容解析成纯文本。

    Args:
        content: 文件字节
        file_type: Java 侧传来的类型，形如 "TXT" / "PDF"

    Raises:
        UnsupportedFileTypeError: 类型暂不支持
    """
    normalized = (file_type or "").strip().upper()

    if normalized not in SUPPORTED_TEXT_TYPES:
        raise UnsupportedFileTypeError(
            f"暂不支持 {normalized} 格式，当前支持的格式："
            f"{' / '.join(sorted(SUPPORTED_TEXT_TYPES))}"
        )

    # 纯文本：先按 UTF-8 解，失败再试 GBK —— 国内用户用 Windows 记事本存的文件
    # 经常是 GBK，直接 utf-8 解会抛 UnicodeDecodeError。
    text = _decode(content)

    # 去掉 BOM 和首尾空白
    text = text.lstrip("\ufeff").strip()

    if not text:
        raise ValueError("文件解析后没有文本内容")

    return ParseResult(text=text, page_count=1)


def _decode(content: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    # 都失败就带替换字符强解，至少不让整个任务挂掉
    logger.warning("文件编码无法识别，使用 utf-8 + errors=replace 强解")
    return content.decode("utf-8", errors="replace")
