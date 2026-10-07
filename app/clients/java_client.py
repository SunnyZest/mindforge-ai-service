"""调用 Java 后端的内部接口。

对应 Java 侧的 InternalDocumentController：

    POST /internal/documents/claim           认领待处理文档
    PUT  /internal/documents/{id}/status     回写处理状态
    POST /internal/documents/{id}/chunks     批量写入分块

设计要点：
· 每个请求都带 X-Service-Token —— Java 那边用 InternalTokenFilter 校验
· 用模块级单例，连接池复用（不要每次请求 new 一个 client）
· URL 显式拼接，不依赖 httpx 的 base_url 合并规则（那个规则容易踩坑）
"""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# 认领/回写很快，但写分块可能几百条，给宽一点
_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=5.0)


class JavaClient:
    """Java 内部接口的客户端。"""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT,
            headers={
                "X-Service-Token": (settings.java_service_token or "").strip(),
                "Content-Type": "application/json",
            },
            # ⚠️ 关键：不要读环境变量里的代理设置。
            #
            # httpx 默认 trust_env=True，会读 HTTP_PROXY / HTTPS_PROXY。
            # 但 Java 后端是【内网/本机】地址，走代理只会出问题 ——
            # 实测踩过：代理把 localhost:8080 的请求也转发出去，
            # 返回 502 "upstream connect failed"，看起来像 Java 挂了，
            # 实际是代理转发失败，排查方向完全被带偏。
            #
            # 需要走代理的是访问公网的那部分（比如 DashScope），
            # 那部分单独建 client 并保持默认即可。
            trust_env=False,
        )

    def _url(self, path: str) -> str:
        """显式拼完整 URL。

        不依赖 httpx 的 base_url 合并规则 —— 那个规则对带路径的 base_url
        （比如 http://localhost:8080/api）处理起来容易出意外，
        自己拼最清楚。
        """
        base = settings.java_base_url.rstrip("/")
        return f"{base}/{path.lstrip('/')}"

    # ---------- ① 认领 ----------

    async def claim(self, limit: int | None = None) -> list[dict]:
        """认领一批待处理文档。

        Java 会把认领到的文档状态改成 PARSING（原子操作，多实例安全），
        所以**同一个文档不会被两个实例同时拿到**。

        返回空列表表示当前没有待处理的任务 —— 这是正常情况，不是错误。
        """
        resp = await self._client.post(
            self._url("/internal/documents/claim"),
            params={"limit": limit or settings.claim_limit},
        )
        self._raise_for_status(resp, "认领文档")
        body = resp.json()
        return body.get("data") or []

    # ---------- ② 回写状态 ----------

    async def update_status(
        self,
        document_id: int | str,
        *,
        parse_status: str | None = None,
        chunk_status: str | None = None,
        vector_status: str | None = None,
        error_message: str | None = None,
        page_count: int | None = None,
    ) -> None:
        """回写处理状态。

        字段传 None 表示「不修改这个字段」—— 所以可以分阶段上报：
        解析完只报 parse_status，分块完再报 chunk_status。

        error_message 例外：传空字符串 "" 表示清空上次的错误。
        """
        payload = {
            "parseStatus": parse_status,
            "chunkStatus": chunk_status,
            "vectorStatus": vector_status,
            "errorMessage": error_message,
            "pageCount": page_count,
        }
        resp = await self._client.put(
            self._url(f"/internal/documents/{document_id}/status"),
            json=payload,
        )
        self._raise_for_status(resp, "回写状态")

    # ---------- ③ 写分块 ----------

    async def save_chunks(self, document_id: int | str, chunks: list[dict]) -> int:
        """批量写入分块，返回写入条数。

        Java 那边是「先按 document_id 物理删、再批量插」，
        所以重复调用是幂等的 —— 不会撞唯一索引。
        """
        resp = await self._client.post(
            self._url(f"/internal/documents/{document_id}/chunks"),
            json={"chunks": chunks},
        )
        self._raise_for_status(resp, "写分块")
        return resp.json().get("data") or 0

    # ---------- 生命周期 ----------

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---------- 内部 ----------

    @staticmethod
    def _raise_for_status(resp: httpx.Response, action: str) -> None:
        """把非 2xx 转成带上下文的异常。

        401 单独提示 —— 这个错误 99% 是两边的 service-token 不一致，
        直接说出来能省掉一轮排查。
        """
        if resp.status_code == 401:
            raise RuntimeError(
                f"{action}失败：401 服务令牌无效。"
                f"请检查 .env 的 JAVA_SERVICE_TOKEN 与 Java 的 "
                f"mindforge.internal.service-token 是否完全一致。"
            )
        if resp.is_success:
            return
        raise RuntimeError(
            f"{action}失败：HTTP {resp.status_code}，响应体={resp.text[:300]}"
        )


# 模块级单例：全局共用一个连接池
java_client = JavaClient()
