"""网页正文提取工具：抓取指定 URL，去除广告与导航，提取高信噪比 Markdown 正文。"""

import asyncio
import ipaddress
import logging
import socket
from urllib.parse import urljoin, urlparse
import httpx
import trafilatura

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# 单页 HTML 最大读取字节数（5 MB），防止超大文件流撑爆内存
_MAX_HTML_BYTES = 5 * 1024 * 1024
_MAX_REDIRECTS = 5
_BLOCKED_CONTENT_PREFIXES = (
    "image/",
    "video/",
    "audio/",
    "application/pdf",
    "application/zip",
    "application/x-rar",
    "application/octet-stream",
)


def _check_safe_hostname_sync(hostname: str) -> None:
    """解析目标域名并拦截环回、私网、链路本地及保留地址（防 SSRF）。"""
    lower = hostname.strip().lower()
    if not lower or lower in {"localhost", "localhost.localdomain"} or lower.endswith((".local", ".internal")):
        raise ValueError("禁止访问本地或内网地址")

    try:
        addr_infos = socket.getaddrinfo(lower, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"无法解析目标域名: {hostname}") from exc

    if not addr_infos:
        raise ValueError(f"无法解析目标域名: {hostname}")

    for info in addr_infos:
        ip_str = info[4][0]
        # 剥离 IPv6 zone index（如 fe80::1%eth0）
        if "%" in ip_str:
            ip_str = ip_str.split("%", 1)[0]
        ip = ipaddress.ip_address(ip_str)
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError("出于安全限制，禁止抓取内网或保留 IP 地址")


async def _validate_public_url(url: str) -> urlparse:
    """校验 URL 协议与目标 IP 安全性。"""
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL 格式不合法，必须以 http:// 或 https:// 开头")

    parsed = urlparse(url)
    if not parsed.netloc or not parsed.hostname:
        raise ValueError("非法的 URL 地址")

    await asyncio.to_thread(_check_safe_hostname_sync, parsed.hostname)
    return parsed


async def _fetch_html_safely(url: str, headers: dict[str, str]) -> tuple[str, str]:
    """带逐跳 SSRF 校验与响应体积上限（5MB）的安全抓取。"""
    current_url = url
    async with httpx.AsyncClient(headers=headers, follow_redirects=False, timeout=25.0) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            await _validate_public_url(current_url)
            async with client.stream("GET", current_url) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        raise RuntimeError("网页重定向缺少 Location 头")
                    current_url = urljoin(current_url, location)
                    continue

                resp.raise_for_status()

                content_type = (resp.headers.get("content-type") or "").lower()
                if any(content_type.startswith(prefix) for prefix in _BLOCKED_CONTENT_PREFIXES):
                    raise ValueError(f"不支持抓取非网页内容类型: {content_type.split(';')[0]}")

                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > _MAX_HTML_BYTES:
                        raise ValueError("目标网页体积超过 5MB 安全上限")
                    chunks.append(chunk)

                raw_bytes = b"".join(chunks)
                encoding = resp.encoding or "utf-8"
                return raw_bytes.decode(encoding, errors="replace"), current_url

    raise RuntimeError("网页重定向次数过多")


def _extract_markdown_sync(html: str, url: str, fallback_title: str) -> tuple[str | None, str]:
    """在工作线程中执行 CPU 密集型正文提取，避免阻塞事件循环。"""
    markdown = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_links=True,
        include_images=False,
        include_tables=True,
        favor_recall=True,
    )
    metadata = trafilatura.extract_metadata(html, default_url=url)
    title = (metadata.title if metadata and metadata.title else fallback_title).strip()
    return markdown, title


async def extract_url(url: str) -> dict:
    """抓取 URL 并提取为结构化 Markdown 正文。"""
    url = (url or "").strip()
    parsed = await _validate_public_url(url)

    logger.info("开始提取网页正文: %s", url)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    try:
        html, final_url = await _fetch_html_safely(url, headers)
    except ValueError:
        raise
    except Exception as exc:
        logger.error("抓取网页失败 [%s]: %s", url, exc)
        raise RuntimeError(f"抓取网页失败: {exc}") from exc

    fallback_title = parsed.path.split("/")[-1] or parsed.netloc
    markdown, title = await asyncio.to_thread(_extract_markdown_sync, html, final_url, fallback_title)

    if not markdown or len(markdown.strip()) < 30:
        raise ValueError("未能从该网页提取到有效正文，可能是纯前端动态渲染页面或存在反爬拦截")

    logger.info("网页提取成功: [%s] 共 %d 字", title, len(markdown))
    return {
        "url": url,
        "title": title,
        "content": markdown.strip(),
        "charCount": len(markdown.strip()),
    }
