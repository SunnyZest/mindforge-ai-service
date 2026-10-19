"""测试 DOCX 解析：普通段落/表格，以及文本框、内容控件这类 python-docx 看不到的位置。

背景：中文简历模板很爱用「文本框」排版，整份简历装在几个 w:txbxContent 里。
python-docx 的 doc.paragraphs / doc.tables 只遍历 body 的直接子元素，这种文件
只能拿到一个空段落，接口最后报「文件解析后没有文本内容」。
"""

import io
import zipfile

from docx import Document

from app.services.parser import parse

# 文本框 + 内容控件的 body 片段。
# mc:Choice 与 mc:Fallback 里放同样的文字 —— 这正是 Word 的实际存储方式
# （新版本读 wps:txbx，老版本读 v:textbox），提取时必须只取一份。
_TEXTBOX_BODY = """
    <w:p/>
    <w:p>
      <w:r>
        <mc:AlternateContent>
          <mc:Choice Requires="wps">
            <w:drawing>
              <wps:txbx>
                <w:txbxContent>
                  <w:p><w:r><w:t>文本框里的姓名：张三</w:t></w:r></w:p>
                  <w:p><w:r><w:t>求职意向：Java 后端</w:t></w:r></w:p>
                </w:txbxContent>
              </wps:txbx>
            </w:drawing>
          </mc:Choice>
          <mc:Fallback>
            <w:pict>
              <v:shape>
                <v:textbox>
                  <w:txbxContent>
                    <w:p><w:r><w:t>文本框里的姓名：张三</w:t></w:r></w:p>
                    <w:p><w:r><w:t>求职意向：Java 后端</w:t></w:r></w:p>
                  </w:txbxContent>
                </v:textbox>
              </v:shape>
            </w:pict>
          </mc:Fallback>
        </mc:AlternateContent>
      </w:r>
    </w:p>
    <w:sectPr/>
"""

_SDT_BODY = """
    <w:sdt>
      <w:sdtPr/>
      <w:sdtContent>
        <w:p><w:r><w:t>内容控件里的段落</w:t></w:r></w:p>
      </w:sdtContent>
    </w:sdt>
    <w:sectPr/>
"""


def _empty_docx() -> bytes:
    buf = io.BytesIO()
    Document().save(buf)
    return buf.getvalue()


def _replace_document_xml(data: bytes, body_xml: str) -> bytes:
    """把 document.xml 换成自定义 body，用于构造 python-docx 造不出来的结构。"""
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        "<w:document"
        ' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
        ' xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'
        ' xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"'
        ' xmlns:v="urn:schemas-microsoft-com:vml"'
        ' mc:Ignorable="wps">'
        f"<w:body>{body_xml}</w:body>"
        "</w:document>"
    )

    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            payload = src.read(item.filename)
            if item.filename == "word/document.xml":
                payload = xml.encode("utf-8")
            dst.writestr(item, payload)
    return out.getvalue()


def test_parse_docx_paragraphs_heading_and_table():
    """常规 docx：段落、标题层级、表格都按原有格式产出。"""
    doc = Document()
    doc.add_heading("教育背景", level=1)
    doc.add_paragraph("二本大学 人工智能工程技术")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "技能"
    table.cell(0, 1).text = "Java"

    buf = io.BytesIO()
    doc.save(buf)

    result = parse(buf.getvalue(), "DOCX")

    assert "# 教育背景" in result.text
    assert "二本大学 人工智能工程技术" in result.text
    assert "技能 | Java" in result.text


def test_parse_docx_textbox_content():
    """文本框排版：正文全在 w:txbxContent 里，doc.paragraphs 取不到。"""
    data = _replace_document_xml(_empty_docx(), _TEXTBOX_BODY)

    result = parse(data, "DOCX")

    assert "文本框里的姓名：张三" in result.text
    assert "求职意向：Java 后端" in result.text


def test_parse_docx_textbox_not_duplicated():
    """mc:Choice 与 mc:Fallback 各存一份同样的文字，提取时只该取一次。"""
    data = _replace_document_xml(_empty_docx(), _TEXTBOX_BODY)

    result = parse(data, "DOCX")

    assert result.text.count("文本框里的姓名：张三") == 1
    assert result.text.count("求职意向：Java 后端") == 1


def test_parse_docx_content_control():
    """w:sdt 包着的段落同样不在 doc.paragraphs 里。"""
    data = _replace_document_xml(_empty_docx(), _SDT_BODY)

    result = parse(data, "DOCX")

    assert "内容控件里的段落" in result.text
