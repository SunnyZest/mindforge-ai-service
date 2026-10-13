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
@pytest.mark.parametrize(
    "ssrf_url",
    [
        "http://localhost:8080/actuator",
        "http://127.0.0.1:8000/internal/ai/v1/health",
        "http://169.254.169.254/latest/meta-data/",
        "http://192.168.1.1/admin",
        "http://10.0.0.1/secret",
    ],
)
async def test_extract_url_blocks_ssrf(ssrf_url: str):
    with pytest.raises(ValueError, match="禁止"):
        await extract_url(ssrf_url)


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

    async def _aiter_bytes():
        yield sample_html.encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
    mock_resp.encoding = "utf-8"
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_bytes = _aiter_bytes

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__.return_value = mock_resp
    mock_stream_ctx.__aexit__.return_value = None

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with (
        patch("app.services.tools.web_extractor._check_safe_hostname_sync", return_value=None),
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        result = await extract_url("https://example.com/java-interview-guide")

    assert result["url"] == "https://example.com/java-interview-guide"
    assert "Java" in result["title"]
    assert "HashMap" in result["content"]
    assert result["charCount"] > 20

