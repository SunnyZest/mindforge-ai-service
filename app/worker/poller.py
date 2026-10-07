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
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)

# 清理频率从配置读（默认 12 轮 ≈ 1 分钟），见 config.py
_PURGE_EVERY_N_ROUNDS = max(1, settings.vector_purge_every_n_rounds)

# 连续失败时的退避上限：Java 挂了不要每 5 秒狂试，把日志刷爆。
#
# 上限刻意压到 30 秒 —— 实测踩过：Java 启动要 10 秒以上，Python 先起来会连不上，
# 如果退避用指数增长（5→10→20→40→60），Java 起来后还要再等 60 秒才恢复，
# 看起来像「Python 坏了」。线性退避最多等 30 秒，体感好很多，
# 又不会在 Java 真挂时把日志刷爆。
_MAX_BACKOFF_SECONDS = 30


async def poll_loop() -> None:
    """主循环：认领 → 处理 → 回写，无任务则休眠。

    由 main.py 的 lifespan 用 asyncio.create_task 启动，
    应用关闭时会被 cancel。
    """
    interval = max(1, settings.poll_interval_seconds)
    consecutive_failures = 0
    rounds = 0

    logger.info("轮询循环已启动（间隔 %d 秒，单次最多认领 %d 篇，每 %d 轮清理一次向量）",
                interval, settings.claim_limit, _PURGE_EVERY_N_ROUNDS)

    while True:
        try:
            tasks = await java_client.claim()

            if tasks:
                # 有任务：退避计数清零
                consecutive_failures = 0
                logger.info("认领到 %d 篇文档：%s",
                            len(tasks), [t.get("fileName") for t in tasks])

                # 串行处理，一次一篇 —— 简单可控，也不会打爆嵌入接口的限流。
                # 单篇文档的处理时间通常远小于轮询间隔，串行不会成为瓶颈。
                for task in tasks:
                    await process_document(task)
            else:
                # 没任务：正常情况，睡一会儿再来
                consecutive_failures = 0
                await asyncio.sleep(interval)

            # 每隔 N 轮清一次已删除文档的向量。
            # 放在这里而不是单独开一个任务：复用同一套退避逻辑，
            # 而且清理失败不会影响认领（_purge_deleted_vectors 内部兜住了异常）。
            rounds += 1
            if rounds >= _PURGE_EVERY_N_ROUNDS:
                rounds = 0
                await _purge_deleted_vectors()

        except asyncio.CancelledError:
            logger.info("轮询循环收到取消信号，正在退出…")
            raise

        except Exception:                            # noqa: BLE001
            # 这里兜的是「连不上 Java」之类的循环级错误，
            # 单个文档的处理失败已经在 process_document 内部消化了。
            consecutive_failures += 1
            # 线性退避（不是指数）—— 见 _MAX_BACKOFF_SECONDS 的注释
            backoff = min(interval * consecutive_failures, _MAX_BACKOFF_SECONDS)
            logger.exception("轮询出错（连续第 %d 次），%d 秒后重试",
                             consecutive_failures, backoff)
            await asyncio.sleep(backoff)


async def _purge_deleted_vectors() -> None:
    """清理「已删除文档」在向量库里留下的向量。

    为什么需要：删除文档只做逻辑删除，向量库不会自动同步 ——
    不清的话会留下「孤儿向量」，检索时可能召回已删文档的内容。

    为什么由本服务主动拉、而不是 Java 主动通知：
    Java 在 Windows、本服务将来可能在 NAT 后的虚拟机里，
    「Java → 本服务」方向需要端口转发，而「本服务 → Java」天然可达。

    这个方法【不抛异常】—— 清理失败不该影响文档认领。
    """
    try:
        docs = await java_client.list_pending_vector_purge()
    except Exception:                                # noqa: BLE001
        logger.exception("拉取待清理向量列表失败（不影响文档处理）")
        return

    if not docs:
        return

    logger.info("开始清理 %d 篇已删除文档的向量", len(docs))
    purged = 0

    for doc in docs:
        document_id = doc.get("documentId")
        try:
            await vector_store.delete_by_document(document_id)
            await java_client.mark_vector_purged(document_id)
            purged += 1
        except Exception:                            # noqa: BLE001
            # 单个失败不影响其他文档；下次轮询还会拿到它重试
            logger.exception("清理文档 %s 的向量失败（下次会重试）", document_id)

    logger.info("向量清理完成：%d/%d 篇", purged, len(docs))
