"""下载文件 —— 文档流水线和简历解析都要用。

Java 侧给的是对象存储的**预签名 URL**，所以本服务不需要配任何 OSS 密钥，
也不该配 —— 访问凭据由 Java 按需签发、带有效期，本服务只当个普通 HTTP 客户端。
"""

import logging

import httpx

logger = logging.getLogger(__name__)

# 文件可能几十 MB，读超时给宽一点
_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


async def download_bytes(url: str) -> bytes:
    """下载一个预签名 URL 指向的文件。

    走默认的 httpx 配置（trust_env=True）—— 对象存储是公网地址，
    本机有代理的话应该走代理。

    Raises:
        httpx.HTTPStatusError: 预签名 URL 过期或无效（常见 403）
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        logger.debug("下载完成：%d 字节", len(resp.content))
        return resp.content
