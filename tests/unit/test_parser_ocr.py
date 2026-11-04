"""测试 PDF 文档解析与扫描件 OCR 降级功能。"""

import io

from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from app.services.parser import ParseResult, parse


def _create_image_pdf(text: str) -> bytes:
    """生成只包含纯图像（无任何文字流）的扫描件型 PDF。"""
    img = Image.new("RGB", (400, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 40), text, fill=(0, 0, 0))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(400, 120))
    c.drawImage(ImageReader(img), 0, 0, 400, 120)
    c.showPage()
    c.save()
    return buf.getvalue()


def test_parse_scanned_pdf_with_ocr():
    """验证扫描件 PDF（图片型 PDF）能够自动触发 OCR 降级提取文本。"""
    pdf_bytes = _create_image_pdf("MindForge OCR Success")
    result: ParseResult = parse(pdf_bytes, "PDF")

    assert result.page_count == 1
    assert "MindForge OCR Success" in result.text
