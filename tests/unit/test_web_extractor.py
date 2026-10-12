import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.tools.web_extractor import extract_url


@pytest.mark.anyio
async def test_extract_url_invalid_scheme():
    with pytest.raises(ValueError, match="URL 格式不合法"):
        await extract_url("ftp://example.com")


@pytest.mark.anyio
async def test_extract_url_empty():
    with pytest.raises(ValueError, match="URL 格式不合法"):
        await extract_url("")


@pytest.mark.anyio
async def test_extract_url_success():
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Java 八股文核心汇总</title></head>
    <body>
        <nav><a href="/">Home</a></nav>
        <article>
            <h1>Java 八股文核心汇总</h1>
            <p>Java 是一门面向对象的编程语言，具有跨平台特性、垃圾回收机制与强大的生态圈。</p>
            <p>在面试中常考的核心知识点包括 HashMap 底层原理、并发编程锁机制、JVM 内存模型等。</p>
        </article>
        <footer>Copyright 2026</footer>
    </body>
    </html>
    """

    mock_resp = MagicMock()
    mock_resp.text = sample_html
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await extract_url("https://example.com/java-interview-guide")

    assert result["url"] == "https://example.com/java-interview-guide"
    assert "Java" in result["title"]
    assert "HashMap" in result["content"]
    assert result["charCount"] > 20
