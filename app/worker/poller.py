"""轮询循环 —— 本服务主动去 Java 那儿领任务。

为什么不反过来（Java 调本服务）：
  Java 在 Windows 本机、本服务将来可能跑在 VM。NAT 网络下
  「VM → Windows」方向天然可达，「Windows → VM」要配端口转发。
  轮询天然规避了这个问题。

将来量大了要换消息队列，只需替换本模块的 _fetch_tasks，
process_document 一个字不用改。
"""

import asyncio
import logging

from app.clients.java_client import java_client
from app.config import settings
from app.services.pipeline import process_document

logger = logging.getLogger(__name__)

# 连续失败时的退避上限：Java 挂了不要每 5 秒狂试，把日志刷爆
_MAX_BACKOFF_SECONDS = 60


async def poll_loop() -> None:
    """主循环：认领 → 处理 → 回写，无任务则休眠。

    由 main.py 的 lifespan 用 asyncio.create_task 启动，
    应用关闭时会被 cancel。
    """
    interval = max(1, settings.poll_interval_seconds)
    consecutive_failures = 0

    logger.info("轮询循环已启动（间隔 %d 秒，单次最多认领 %d 篇）",
                interval, settings.claim_limit)

    while True:
        try:
            tasks = await java_client.claim()

            if not tasks:
                # 没任务：正常情况，睡一会儿再来
                consecutive_failures = 0
                await asyncio.sleep(interval)
                continue

            # 有任务：退避计数清零
            consecutive_failures = 0
            logger.info("认领到 %d 篇文档：%s",
                        len(tasks), [t.get("fileName") for t in tasks])

            # 串行处理 —— 第 1 步先求简单可控。
            # 文档多了以后再改成有上限的并发（注意别打爆 DashScope 的限流）。
            for task in tasks:
                await process_document(task)

            # 处理完不 sleep，立刻回上去领下一批 —— 可能还有

        except asyncio.CancelledError:
            logger.info("轮询循环收到取消信号，正在退出…")
            raise

        except Exception:                            # noqa: BLE001
            # 这里兜的是「连不上 Java」之类的循环级错误，
            # 单个文档的处理失败已经在 process_document 内部消化了。
            consecutive_failures += 1
            backoff = min(interval * (2 ** min(consecutive_failures, 4)), _MAX_BACKOFF_SECONDS)
            logger.exception("轮询出错（连续第 %d 次），%.0f 秒后重试",
                             consecutive_failures, backoff)
            await asyncio.sleep(backoff)
