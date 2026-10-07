"""MindForge AI Service —— FastAPI 入口。

第 0 步只做「把管子接上」：
  · 起一个 HTTP 服务（/health）
  · 能调通 Java 的内部接口（启动时探测一次 + /health/java 手动探测）
  · 还不处理任何文档

启动：
    uv run uvicorn app.main:app --reload --port 8000
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health
from app.clients.java_client import java_client
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# httpx 默认每次请求都打一条 INFO。
# 但轮询是每 5 秒一次的空转请求 —— 一天能刷出十几 MB 无意义日志，
# 真正有用的业务日志会被淹掉。调到 WARNING，只在出错时说话。
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def _probe_java() -> None:
    """启动时探测一次 Java —— 只打日志，不阻塞启动。

    用 create_task 而不是 await：Java 没起的时候不能把本服务卡死。
    """
    try:
        docs = await java_client.claim(limit=1)
        logger.info("✅ 已连上 Java（%s），当前待处理文档 %d 篇",
                    settings.java_base_url, len(docs))
        if docs:
            logger.info("   认领到：%s", [d.get("fileName") for d in docs])
            logger.info("   （第 0 步还没实现处理逻辑，这些文档会被 Java 的超时任务回收）")
    except Exception as exc:                       # noqa: BLE001
        logger.error("❌ 连不上 Java：%s", exc)
        logger.error("   请检查：① Java 是否已启动  ② JAVA_SERVICE_TOKEN 是否与 Java 一致")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("MindForge AI Service 启动中…")
    logger.info("   Java 后端    : %s", settings.java_base_url)
    logger.info("   分块参数     : chunk_size=%d, overlap=%d",
                settings.chunk_size, settings.chunk_overlap)

    # 启动时探测一次 Java（只打日志，不阻塞启动 —— Java 没起也不能把本服务卡死）
    probe_task = asyncio.create_task(_probe_java())

    yield

    logger.info("MindForge AI Service 关闭中…")
    probe_task.cancel()
    await java_client.aclose()


app = FastAPI(
    title="MindForge AI Service",
    description="文档解析、分块、向量化、RAG 检索与对话 —— 只给 Java 后端调用",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
