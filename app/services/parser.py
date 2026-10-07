"""文档解析：把文件字节变成纯文本。

支持的类型：TXT / MD / PDF / DOCX / HTML。

统一产出【带结构提示的纯文本】—— 标题保留 `#` 前缀，表格转成 `|` 分隔的行。
这样后面分块时能识别结构边界，检索回来的片段也更容易读懂。

不支持的类型（比如老的 .doc、扫描件 PDF）会明确报错，
而不是静默产出一堆空字符串或者乱码 —— 报错好过脏数据。
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 各类型的处理函数在文件末尾注册
SUPPORTED_TYPES = ("TXT", "MD", "MARKDOWN", "PDF", "DOCX", "HTML", "HTM")


class UnsupportedFileTypeError(Exception):
    """文件类型不支持。"""


class ParseError(Exception):
    """文件类型支持，但内容解析不出来（加密、损坏、扫描件等）。"""


@dataclass
class ParseResult:
    text: str
    # 页数。只有 PDF 有这个概念；DOCX/HTML/TXT 是流式文档，返回 None
    page_count: int | None = None


def parse(content: bytes, file_type: str) -> ParseResult:
    """把文件内容解析成纯文本。

    Args:
        content: 文件字节
        file_type: Java 侧传来的类型，形如 "TXT" / "PDF"

    Raises:
        UnsupportedFileTypeError: 类型不支持
        ParseError: 类型支持但解析失败
    """
    normalized = (file_type or "").strip().upper()

    handler = _HANDLERS.get(normalized)
    if handler is None:
        raise UnsupportedFileTypeError(
            f"暂不支持 {normalized or '未知'} 格式，当前支持的格式："
            f"{' / '.join(SUPPORTED_TYPES)}"
        )

    result = handler(content)

    # 统一清理：去 BOM、压缩多余空行
    text = _normalize(result.text)
    if not text:
        raise ParseError("文件解析后没有文本内容")

    return ParseResult(text=text, page_count=result.page_count)


# ---------- 纯文本 ----------


def _parse_text(content: bytes) -> ParseResult:
    """TXT / MD —— 直接解码。

    中文 Windows 记事本存的文件常是 GBK，直接 utf-8 解会抛 UnicodeDecodeError，
    所以按 utf-8 → utf-8-sig → gbk 的顺序试。
    """
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return ParseResult(text=content.decode(encoding), page_count=None)
        except UnicodeDecodeError:
            continue

    logger.warning("文件编码无法识别，使用 utf-8 + errors=replace 强解")
    return ParseResult(text=content.decode("utf-8", errors="replace"), page_count=None)


# ---------- PDF ----------


def _parse_pdf(content: bytes) -> ParseResult:
    """PDF —— 用 pdfplumber 提取文本。

    为什么用 pdfplumber 而不是 pypdf：实测中文 PDF 下 pdfplumber 更快
    （0.47s vs 0.99s）且提取结果一样完整，对复杂排版的容忍度也更好。
    pypdf 留作兜底 —— PDF 格式太杂，某些畸形文件 pdfplumber 会失败。
    """
    try:
        return _parse_pdf_plumber(content)
    except ParseError:
        raise
    except Exception as exc:                          # noqa: BLE001
        logger.warning("pdfplumber 提取失败（%s），改用 pypdf 重试", exc)

    return _parse_pdf_pypdf(content)


def _parse_pdf_plumber(content: bytes) -> ParseResult:
    import io

    import pdfplumber

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        pages = [(page.extract_text() or "") for page in pdf.pages]

    text = "\n\n".join(p for p in pages if p.strip())
    if not text.strip():
        raise ParseError(
            "PDF 里没有提取到任何文本 —— 大概率是扫描件（图片型 PDF），需要 OCR 才能处理。"
            "如果是加密 PDF，也请先去掉密码。"
        )

    return ParseResult(text=text, page_count=len(pages))


def _parse_pdf_pypdf(content: bytes) -> ParseResult:
    import io

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise ParseError("PDF 已加密，请先去掉密码再上传")
        pages = [(p.extract_text() or "") for p in reader.pages]
    except ParseError:
        raise
    except Exception as exc:                          # noqa: BLE001
        raise ParseError(f"PDF 解析失败（文件可能已损坏）：{exc}") from exc

    text = "\n\n".join(p for p in pages if p.strip())
    if not text.strip():
        raise ParseError(
            "PDF 里没有提取到任何文本 —— 大概率是扫描件（图片型 PDF），需要 OCR 才能处理。"
        )

    return ParseResult(text=text, page_count=len(pages))


# ---------- DOCX ----------


def _parse_docx(content: bytes) -> ParseResult:
    """DOCX —— 段落 + 表格。

    ⚠️ python-docx 只能读 .docx（Office 2007+ 的 XML 格式），
    读不了老的 .doc（二进制格式）。那个类型应该在 Java 侧就被拦掉。
    """
    import io

    from docx import Document

    try:
        doc = Document(io.BytesIO(content))
    except Exception as exc:                          # noqa: BLE001
        raise ParseError(
            f"DOCX 解析失败（文件可能已损坏，或者是老的 .doc 格式）：{exc}"
        ) from exc

    parts: list[str] = []

    # 段落：标题转成 # 前缀，保留层级信息
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if not t:
            continue
        parts.append(_decorate_heading(p.style.name if p.style else "", t))

    # 表格：python-docx 的 paragraphs 不含表格内容，得单独取。
    # 转成 `a | b | c` 的行 —— 分块时能看出这是表格数据。
    for table in doc.tables:
        for row in table.rows:
            cells = [(c.text or "").strip() for c in row.cells]
            if any(cells):                            # 跳过整行空的
                parts.append(" | ".join(cells))

    return ParseResult(text="\n".join(parts), page_count=None)


def _decorate_heading(style_name: str, text: str) -> str:
    """把 Word 的标题样式转成 Markdown 的 # 前缀。"""
    if style_name.startswith("Heading"):
        level = style_name.replace("Heading", "").strip()
        if level.isdigit() and 1 <= int(level) <= 6:
            return f"{'#' * int(level)} {text}"
    if style_name in ("Title", "标题"):
        return f"# {text}"
    return text


# ---------- HTML ----------


# 块级标签 —— 它们前后要断行。行内标签（code/strong/em/a 等）不断行，
# 否则「自动配置通过 <code>xxx</code> 开启」会被拆成三行，句子就断了。
_BLOCK_TAGS = (
    "address", "article", "aside", "blockquote", "dd", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3",
    "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
    "section", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
)


def _parse_html(content: bytes) -> ParseResult:
    """HTML —— 去标签、保留结构。

    要干掉 script / style / 注释，否则代码和 CSS 会混进正文污染检索。
    """
    from bs4 import BeautifulSoup, Comment

    try:
        soup = BeautifulSoup(_decode_for_html(content), "lxml")
    except Exception as exc:                          # noqa: BLE001
        raise ParseError(f"HTML 解析失败：{exc}") from exc

    # 剔除不该进正文的内容
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # 标题先加 # 前缀（在插换行之前做，这时候 get_text 还没被换行污染）
    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            heading = h.get_text(strip=True)
            if heading:
                h.clear()
                h.append(f"{'#' * level} {heading}")

    # 块级标签前后插换行；行内标签不动，保证句子完整
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    return ParseResult(text=soup.get_text(), page_count=None)


def _decode_for_html(content: bytes) -> str:
    """HTML 的编码更杂，先看 meta 声明，再按常见编码试。"""
    head = content[:2048].lower()
    if b"charset=gbk" in head or b"charset=gb2312" in head:
        return content.decode("gbk", errors="replace")
    return _parse_text(content).text


# ---------- 内部 ----------


def _normalize(text: str) -> str:
    """统一清理：去 BOM、去行尾空白、压缩连续空行。"""
    lines = [line.rstrip() for line in text.lstrip("\ufeff").splitlines()]

    cleaned: list[str] = []
    blank = 0
    for line in lines:
        if line.strip():
            blank = 0
            cleaned.append(line)
        else:
            blank += 1
            # 连续空行最多保留一个 —— 分块时用 \n\n 当段落边界，
            # 一堆空行会把边界判断搞乱
            if blank == 1:
                cleaned.append("")

    return "\n".join(cleaned).strip()


_HANDLERS = {
    "TXT": _parse_text,
    "MD": _parse_text,
    "MARKDOWN": _parse_text,
    "PDF": _parse_pdf,
    "DOCX": _parse_docx,
    "HTML": _parse_html,
    "HTM": _parse_html,
}
