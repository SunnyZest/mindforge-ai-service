"""服务间认证。

两个方向，别搞混：

  Java → Python   走本模块的 verify_service_token（校验 Java 带来的 X-Service-Token）
  Python → Java   走 app/clients/java_client.py（主动带上 X-Service-Token）

本服务对外提供 /internal/ai/v1/** 时用第一个方向校验调用方。
"""

import logging

from fastapi import Header, HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)


async def verify_service_token(
    x_service_token: str | None = Header(default=None, alias="X-Service-Token"),
) -> None:
    """FastAPI 依赖：校验请求头里的服务令牌。

    用法：

        @router.post("/documents/parse", dependencies=[Depends(verify_service_token)])
        async def parse(...): ...

    fail-safe：令牌没配置时**拒绝所有请求**，而不是放行 ——
    配置漏了应该「用不了」，而不是「裸奔」。
    """
    expected = (settings.java_service_token or "").strip()

    if not expected:
        logger.error("收到内部请求，但 JAVA_SERVICE_TOKEN 未配置，已拒绝")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="服务未配置令牌",
        )

    if (x_service_token or "").strip() != expected:
        logger.warning("服务令牌校验失败")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="服务令牌无效",
        )
