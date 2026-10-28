"""文档解析：把文件字节变成纯文本。

支持的类型：TXT / MD / PDF / DOCX / HTML / PNG / JPG。

统一产出【带结构提示的纯文本】—— 标题保留 `#` 前缀，表格转成 `|` 分隔的行。
这样后面分块时能识别结构边界，检索回来的片段也更容易读懂。

不支持的类型（比如老的 .doc、扫描件 PDF）会明确报错，
而不是静默产出一堆空字符串或者乱码 —— 报错好过脏数据。

⚠️ 图片（PNG / JPG）是 2026-10-28 为「面试邀约截图」加的，走本地 RapidOCR。
   它<不是>知识库文档类型：文档上传的类型白名单在 Java 侧，和这里无关。
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 各类型的处理函数在文件末尾注册
SUPPORTED_TYPES = ("TXT", "MD", "MARKDOWN", "PDF", "DOCX", "HTML", "HTM", "PNG", "JPG", "JPEG")


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


# 缓存 RapidOCR 实例，避免单次任务重复初始化模型
_rapid_ocr_instance = None


def _get_rapid_ocr():
    global _rapid_ocr_instance
    if _rapid_ocr_instance is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _rapid_ocr_instance = RapidOCR()
        except Exception as exc:  # noqa: BLE001
            logger.error("RapidOCR 初始化失败: %s", exc)
            raise ParseError(f"本地 OCR 引擎加载失败: {exc}") from exc
    return _rapid_ocr_instance


def _ocr_page_dashscope(image_bytes: bytes) -> str:
    """尝试使用 DashScope 视觉多模态大模型进行高精度排版 OCR。"""
    try:
        from app.config import settings
        from app.services import ai_config

        cfg = ai_config.get(ai_config.SCENE_CHAT)
        api_key = cfg.api_key or settings.dashscope_api_key
        base_url = cfg.base_url or settings.dashscope_base_url

        if not api_key:
            return ""

        import base64
        from openai import OpenAI

        base64_img = base64.b64encode(image_bytes).decode("utf-8")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0,
            max_retries=1,
        )
        response = client.chat.completions.create(
            model="qwen-vl-ocr",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请提取图片中的全部文字。保持原有的排版、标题层级与段落结构，遇到表格请使用 Markdown 语法还原表格，直接输出识别结果，不要输出任何额外的解释或问候。",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                        },
                    ],
                }
            ],
            temperature=0.01,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("DashScope OCR 调用跳过或失败: %s", exc)
        return ""


def _ocr_page_rapidocr(image) -> str:
    """使用本地 RapidOCR (ONNX) 离线识别单页图片。"""
    import numpy as np

    ocr = _get_rapid_ocr()
    result, _ = ocr(np.array(image))
    if not result:
        return ""
    return "\n".join(item[1] for item in result if item and len(item) > 1 and item[1]).strip()


# ---------- 图片（截图） ----------


def _parse_image(content: bytes) -> ParseResult:
    """PNG / JPG —— 本地 RapidOCR 离线识别。

    ⚠️ 这里刻意**不**像 PDF 那样先试 DashScope 多模态，两个原因：
      1. 截图里是聊天记录 / 邮件，要的是「按行排列的原文」。
         RapidOCR 给的正是逐行文本；多模态大模型常会重排、补标点、改写，
         一旦它改了字，下游的 evidence 子串校验（原文.contains(片段)）就会整片失败，
         用户看到的是「字段全被丢了」而不是「识别不准」。
      2. 本地引擎没有网络成本和延迟，一张截图 1~2 秒。
    """
    import io

    from PIL import Image

    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"图片无法读取（可能已损坏）：{exc}") from exc

    # 带透明通道的 PNG 直接喂进去会得到 4 通道，OCR 会报错；统一转 RGB
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    try:
        text = _ocr_page_rapidocr(image)
    except ParseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"图片 OCR 识别失败：{exc}") from exc

    if not text.strip():
        raise ParseError("图片里没有识别到文字 —— 可能截图太模糊，或者图里不是文字")

    return ParseResult(text=text, page_count=1)


def _parse_pdf_ocr(content: bytes) -> ParseResult:
    """PDF OCR 降级处理：针对纯图片型扫描件。

    优先尝试 DashScope 多模态 OCR（带表格与排版还原）；
    若无 API Key 或网络调用失败，无缝回退至本地 RapidOCR 离线引擎。
    """
    import io
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(content)
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"PDF 文件损坏，无法读取页面: {exc}") from exc

    total_pages = len(pdf)
    if total_pages == 0:
        raise ParseError("PDF 文件没有任何页面")

    # 防刷：单文档最多识别前 30 页
    max_pages = min(total_pages, 30)
    page_texts: list[str] = []

    for i in range(max_pages):
        page = pdf[i]
        pil_image = page.render(scale=2.0).to_pil()

        page_text = ""
        # 1. 优先尝试云端多模态 OCR
        try:
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            page_text = _ocr_page_dashscope(buf.getvalue())
        except Exception as exc:  # noqa: BLE001
            logger.warning("第 %d 页 DashScope OCR 失败（%s），降级至本地 RapidOCR", i + 1, exc)

        # 2. 回退本地 RapidOCR
        if not page_text:
            try:
                page_text = _ocr_page_rapidocr(pil_image)
            except Exception as exc:  # noqa: BLE001
                logger.warning("第 %d 页 本地 RapidOCR 识别失败: %s", i + 1, exc)

        if page_text.strip():
            page_texts.append(f"<!-- Page {i + 1} -->\n{page_text.strip()}")

    text = "\n\n".join(page_texts).strip()
    if not text:
        raise ParseError("PDF 扫描件经 OCR 处理后未能识别出任何文字内容")

    return ParseResult(text=text, page_count=total_pages)


def _parse_pdf(content: bytes) -> ParseResult:
    """PDF —— 优先用 pdfplumber 提取原生文本，若为扫描件/图片型 PDF 则自动降级 OCR。

    为什么用 pdfplumber 而不是 pypdf：实测中文 PDF 下 pdfplumber 更快
    （0.47s vs 0.99s）且提取结果一样完整，对复杂排版的容忍度也更好。
    pypdf 留作兜底；扫描件自动走 OCR 识别。
    """
    try:
        return _parse_pdf_plumber(content)
    except ParseError as exc:
        if "扫描件" in str(exc):
            logger.info("pdfplumber 未提取到文本，切换至 OCR 引擎: %s", exc)
            return _parse_pdf_ocr(content)
        raise
    except Exception as exc:                          # noqa: BLE001
        logger.warning("pdfplumber 提取失败（%s），改用 pypdf 重试", exc)

    try:
        return _parse_pdf_pypdf(content)
    except ParseError as exc:
        if "扫描件" in str(exc):
            logger.info("pypdf 未提取到文本，切换至 OCR 引擎: %s", exc)
            return _parse_pdf_ocr(content)
        raise


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

# WordprocessingML 与 markup-compatibility 的命名空间。
# 下面要直接读 XML —— python-docx 的 paragraphs/tables 看不到文本框里的内容。
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_MC_NS = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"


def _parse_docx(content: bytes) -> ParseResult:
    """DOCX —— 段落 + 表格 + 文本框。

    ⚠️ python-docx 只能读 .docx（Office 2007+ 的 XML 格式），
    读不了老的 .doc（二进制格式）。那个类型应该在 Java 侧就被拦掉。

    ⚠️ doc.paragraphs / doc.tables 只遍历 body 的直接子元素。中文简历模板
    很爱用「文本框」排版 —— 整份简历装在几个 w:txbxContent 里，这种文件
    doc.paragraphs 只能拿到一个空段落，最后就报「文件解析后没有文本内容」。
    所以下面额外做一遍深度扫描，把文本框与内容控件里的段落也取出来。
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

    # 文本框 / 内容控件：上面两个 API 看不到的地方，单独补一遍。
    # 追加在末尾 —— 混排文档里文本框内容会排在正文之后，但不会丢。
    parts.extend(_iter_extra_paragraph_texts(doc.element.body))

    return ParseResult(text="\n".join(parts), page_count=None)


def _iter_extra_paragraph_texts(node, hidden: bool = False):
    """产出 python-docx 取不到的段落文字：文本框(w:txbxContent) / 内容控件(w:sdt) 里的。

    ⚠️ 必须跳过 mc:Fallback —— 那是 Word 给老版本留的降级副本，同一段文字在
    mc:Choice 的 wps:txbx 与 mc:Fallback 的 v:textbox 里各存一份，
    不跳的话整份正文会重复一遍。
    """
    for child in node:
        tag = child.tag
        if tag == _MC_NS + "Fallback":
            continue
        if tag == _W_NS + "p":
            if hidden:
                text = _paragraph_text(child)
                if text:
                    yield text
            # 段落里可能挂着文本框，继续往下走
            yield from _iter_extra_paragraph_texts(child, hidden)
        elif tag == _W_NS + "tbl":
            # 顶层表格由 doc.tables 负责（那边能拼出 `|`），这里只捞文本框里的
            yield from _iter_extra_paragraph_texts(child, hidden)
        else:
            yield from _iter_extra_paragraph_texts(
                child, hidden or tag in (_W_NS + "txbxContent", _W_NS + "sdt")
            )


def _paragraph_text(p) -> str:
    """取段落自身的文字，不吞掉嵌套段落（文本框里那层）的文字。"""
    buf: list[str] = []

    def walk(n):
        for c in n:
            if c.tag == _W_NS + "p":
                continue                              # 嵌套段落由外层递归单独处理
            if c.tag == _W_NS + "t":
                buf.append(c.text or "")
            elif c.tag == _W_NS + "tab":
                buf.append("\t")
            elif c.tag in (_W_NS + "br", _W_NS + "cr"):
                buf.append("\n")
            else:
                walk(c)

    walk(p)
    return "".join(buf).strip()


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
    "PNG": _parse_image,
    "JPG": _parse_image,
    # 前端会把 JPEG 归一成 JPG 再传，这里兜一下别的客户端
    "JPEG": _parse_image,
}
