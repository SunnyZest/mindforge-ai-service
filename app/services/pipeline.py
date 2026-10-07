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
from app.services.embedding import embedding_service
from app.services.parser import ParseError, UnsupportedFileTypeError, parse
from app.services.splitter import estimate_tokens, split
from app.services.vector_store import vector_store

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

        # 防御性上限：超大文件可能切出几万块，把 embedding API 打爆（也烧钱）
        if len(pieces) > settings.max_chunks_per_document:
            raise ValueError(
                f"分块数 {len(pieces)} 超过上限 {settings.max_chunks_per_document}，"
                f"请拆分后再上传"
            )

        logger.info("  ③ 分块完成：%d 个片段（size=%d, overlap=%d）",
                    len(pieces), settings.chunk_size, settings.chunk_overlap)

        # 分块确实成功了，先如实上报 —— 后面向量化失败不该让这个状态变成 FAILED
        await java_client.update_status(document_id, chunk_status="SUCCESS")

        # ---------- ④ 向量化 ----------
        # ⚠️ 这里是数据隔离的关键点：把 knowledge_base_id 写进向量 metadata，
        #    检索时靠它过滤。Java 的 claim 响应里带了这个字段，必须原样传下去。
        vector_ids: list[str] | None = None
        try:
            embeddings = await embedding_service.embed(pieces)
            vector_ids = await vector_store.add(
                document_id=document_id,
                knowledge_base_id=task["knowledgeBaseId"],
                user_id=task["userId"],
                chunks=pieces,
                embeddings=embeddings,
            )
            logger.info("  ④ 向量化完成：%d 条（kb=%s）",
                        len(vector_ids), task["knowledgeBaseId"])
        except Exception:                            # noqa: BLE001
            # 向量化失败【不往外抛】—— 分块结果仍然要保存，
            # 否则用户看到 chunk_status=SUCCESS 但一条分块都没有，更困惑。
            logger.exception("  ④ 向量化失败（分块结果仍会保存，但这些分块不会被检索到）")

        # ---------- ⑤ 写回 Java ----------
        chunks = [
            {
                "chunkIndex": i,
                "content": piece,
                "tokenCount": estimate_tokens(piece),
                "pageNo": 1,
                "vectorId": vector_ids[i] if vector_ids else None,
                "metadataJson": None,
            }
            for i, piece in enumerate(pieces)
        ]
        inserted = await java_client.save_chunks(document_id, chunks)

        await java_client.update_status(
            document_id,
            vector_status="SUCCESS" if vector_ids else "FAILED",
            error_message=None if vector_ids else "向量化失败，详见 AI 服务日志",
        )

        if vector_ids:
            logger.info("  ⑤ 完成：写入 %d 个分块 + %d 条向量", inserted, len(vector_ids))
            return True

        logger.warning("  ⑤ 部分完成：写入 %d 个分块，但向量化失败", inserted)
        return False

    except (UnsupportedFileTypeError, ParseError) as exc:
        # 这两类属于「预期内的失败」（类型不支持 / 扫描件 / 文件损坏），
        # 日志用 warning 不用 exception —— 打一堆堆栈只会淹没真正的问题
        logger.warning("  文档 %s 处理失败：%s", document_id, exc)
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
