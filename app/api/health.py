"""健康检查。

/health         本服务状态（不碰 Java，适合做容器探针）
/health/java    探测 Java 连通性 + 令牌是否有效（手动排查用）
"""

import logging

from fastapi import APIRouter

from app.clients.java_client import java_client
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["健康检查"])


@router.get("/health")
async def health() -> dict:
    """本服务是否活着。不访问 Java —— 探针应该快且无副作用。"""
    return {
        "status": "ok",
        "service": "mindforge-ai-service",
        "pollerEnabled": settings.enable_poller,
        "pollIntervalSeconds": settings.poll_interval_seconds,
    }


@router.get("/health/java")
async def health_java() -> dict:
    """探测与 Java 的连通性。

    ⚠️ **有副作用**：它会真的调一次 claim，如果当时有待处理文档，会被认领走
    （30 分钟后 Java 的超时任务会回收）。所以这个接口只用于手动排查，
    不要配成监控探针 —— 监控用 /health。
    """
    try:
        docs = await java_client.claim(limit=1)
    except Exception as exc:                       # noqa: BLE001 - 这里就是要兜住所有错
        logger.warning("探测 Java 失败：%s", exc)
        return {
            "status": "error",
            "javaBaseUrl": settings.java_base_url,
            "error": str(exc),
        }

    return {
        "status": "ok",
        "javaBaseUrl": settings.java_base_url,
        "claimed": len(docs),
        "note": "claimed=0 表示连通且无待处理文档；能返回就说明令牌是对的",
    }
