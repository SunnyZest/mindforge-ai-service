"""网页正文提取工具：抓取指定 URL，去除广告与导航，提取高信噪比 Markdown 正文。"""

import logging
from urllib.parse import urlparse
import httpx
import trafilatura

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


async def extract_url(url: str) -> dict:
    """抓取 URL 并提取为结构化 Markdown 正文。

    Args:
        url: 网页完整地址

    Returns:
        {
            "url": str,
            "title": str,
            "content": str,      # 纯净的 Markdown 正文
            "charCount": int,
        }

    Raises:
        ValueError: URL 不合法或未能提取到正文
        RuntimeError: 网络请求失败
    """
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL 格式不合法，必须以 http:// 或 https:// 开头")

    parsed = urlparse(url)
    if not parsed.netloc:
        raise ValueError("非法的 URL 地址")

    logger.info("开始提取网页正文: %s", url)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    try:
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=25.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        logger.error("抓取网页失败 [%s]: %s", url, exc)
        raise RuntimeError(f"抓取网页失败: {exc}") from exc

    # 提取正文并转换为 Markdown
    markdown = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_links=True,
        include_images=False,
        include_tables=True,
        favor_recall=True,
    )

    if not markdown or len(markdown.strip()) < 30:
        raise ValueError("未能从该网页提取到有效正文，可能是纯前端动态渲染页面或存在反爬拦截")

    # 提取标题
    metadata = trafilatura.extract_metadata(html, default_url=url)
    title = (metadata.title if metadata and metadata.title else parsed.path.split("/")[-1] or parsed.netloc).strip()

    logger.info("网页提取成功: [%s] 共 %d 字", title, len(markdown))
    return {
        "url": url,
        "title": title,
        "content": markdown.strip(),
        "charCount": len(markdown.strip()),
    }
