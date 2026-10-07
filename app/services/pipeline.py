"""★ 文档处理流水线。

核心设计：process_document 是一个【纯函数】——
它只关心「处理这一个文档」，完全不关心任务是从哪来的。

  现在：轮询循环调用它
  将来：换成消息队列，只改「怎么拿到任务」，这个文件一个字不动

所以别在这里写任何「怎么获取任务」的逻辑。
"""

import logging

import httpx

from app.clients.java_client import java_client
from app.config import settings
from app.services.parser import UnsupportedFileTypeError, parse
from app.services.splitter import estimate_tokens, split

logger = logging.getLogger(__name__)

# 下载文件的超时：文档可能几十 MB，给宽一点
_DOWNLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


async def process_document(task: dict) -> bool:
    """处理单个文档：下载 → 解析 → 分块 → 回写状态。

    Args:
        task: Java 的 /claim 返回的元素，含 documentId / fileName / fileType /
              downloadUrl / knowledgeBaseId / userId

    Returns:
        True 表示处理成功；False 表示失败（失败原因已回写给 Java）

    注意：这个函数【不抛异常】—— 所有错误都在内部兜住并回写 FAILED。
    因为调用方（轮询循环）不该因为一个文档失败就中断整批任务。
    """
    document_id = task.get("documentId")
    file_name = task.get("fileName")
    file_type = task.get("fileType")

    logger.info("─" * 60)
    logger.info("开始处理文档 %s（%s，类型 %s）", document_id, file_name, file_type)

    try:
        # ---------- ① 下载 ----------
        # Java 给的是 OSS 的预签名 URL，所以本服务不需要配任何 OSS 密钥。
        # 走默认的 httpx（trust_env=True）—— OSS 是公网地址，有代理的话应该走代理。
        content = await _download(task["downloadUrl"])
        logger.info("  ① 下载完成：%d 字节", len(content))

        # ---------- ② 解析 ----------
        result = parse(content, file_type)
        logger.info("  ② 解析完成：%d 字符", len(result.text))

        await java_client.update_status(
            document_id,
            parse_status="SUCCESS",
            page_count=result.page_count,
        )

        # ---------- ③ 分块 ----------
        pieces = split(result.text, settings.chunk_size, settings.chunk_overlap)
        if not pieces:
            raise ValueError("分块后没有任何内容")

        logger.info("  ③ 分块完成：%d 个片段（size=%d, overlap=%d）",
                    len(pieces), settings.chunk_size, settings.chunk_overlap)

        # ---------- ④ 写回 Java ----------
        chunks = [
            {
                "chunkIndex": i,
                "content": piece,
                "tokenCount": estimate_tokens(piece),
                "pageNo": 1,
                "vectorId": None,          # 第 1 步不做向量化
                "metadataJson": None,
            }
            for i, piece in enumerate(pieces)
        ]
        inserted = await java_client.save_chunks(document_id, chunks)

        await java_client.update_status(
            document_id,
            chunk_status="SUCCESS",
            # vector_status 保持 PENDING —— 第 2 步才做向量化。
            # 这里刻意不设成 SUCCESS：假的状态会让后面的检索把没向量的文档也捞出来。
        )

        logger.info("  ④ 完成：写入 %d 个分块（vector_status 仍为 PENDING，等第 2 步）",
                    inserted)
        return True

    except UnsupportedFileTypeError as exc:
        # 类型不支持属于「预期内的失败」，日志用 warning 不用 exception
        logger.warning("  文档 %s 处理失败（类型不支持）：%s", document_id, exc)
        await _mark_failed(document_id, str(exc))
        return False

    except Exception as exc:                       # noqa: BLE001
        logger.exception("  文档 %s 处理失败", document_id)
        await _mark_failed(document_id, f"{type(exc).__name__}: {exc}")
        return False


# ---------- 内部 ----------


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


async def _mark_failed(document_id, message: str) -> None:
    """把失败原因回写给 Java。

    单独包一层是为了「回写本身失败」不会掩盖原始错误 ——
    回写失败只记日志，不再往外抛。
    """
    try:
        await java_client.update_status(
            document_id,
            parse_status="FAILED",
            error_message=message[:500],        # 别把超长堆栈塞进数据库
        )
    except Exception:                                # noqa: BLE001
        logger.exception("回写失败状态时出错（原始错误：%s）", message)
