"""Chroma 向量库 —— ★ 多知识库隔离在这里。

╔══════════════════════════════════════════════════════════════════════╗
║  ⚠️ 安全边界，改动前先读完这一段                                      ║
║                                                                      ║
║  本项目是【多用户 × 多知识库】。所有向量存在同一个 collection 里，      ║
║  靠 metadata 区分归属。检索时如果不带 knowledge_base_id 过滤，        ║
║  用户 A 就能召回用户 B 的文档内容 —— 这是数据泄漏，不是小 bug。        ║
║                                                                      ║
║  所以 query() 【强制】要求 knowledge_base_id，空值直接抛异常。        ║
║  不靠「调用方记得传」，靠「不传就报错」。                              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import chromadb

from app.config import settings

logger = logging.getLogger(__name__)

# app/services/vector_store.py → parents[2] 就是项目根
# 用绝对路径，避免「从哪个目录启动」影响向量库位置
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class VectorStore:
    """Chroma 向量库封装。

    chromadb 是同步 API，在 async 环境里直接调用会阻塞事件循环，
    所以所有操作都包了 asyncio.to_thread。
    """

    def __init__(self) -> None:
        path = _PROJECT_ROOT / settings.chroma_path
        path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            # cosine 距离 —— 文本嵌入的常规选择
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("向量库就绪：%s（collection=%s，现有 %d 条）",
                    path, settings.chroma_collection, self._collection.count())

    # ---------- 写入 ----------

    async def add(
        self,
        *,
        document_id: int | str,
        knowledge_base_id: int | str,
        user_id: int | str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> list[str]:
        """写入一个文档的所有分块向量，返回 vector_id 列表。

        幂等：内部先删掉这个 document_id 的旧向量再写 ——
        重试或重复消费不会产生重复向量（Chroma 没有唯一约束，只能自己保证）。
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"分块数({len(chunks)})和向量数({len(embeddings)})不一致"
            )
        if not chunks:
            return []

        doc_id = int(document_id)
        kb_id = int(knowledge_base_id)
        uid = int(user_id)

        # 幂等：先清旧的
        await self.delete_by_document(doc_id)

        ids = [f"doc{doc_id}-chunk{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": doc_id,
                "knowledge_base_id": kb_id,   # ★ 检索时靠它过滤
                "user_id": uid,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        def _write() -> None:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

        await asyncio.to_thread(_write)
        logger.info("向量写入完成：documentId=%s，%d 条（kb=%s）", doc_id, len(ids), kb_id)
        return ids

    # ---------- 检索 ----------

    async def query(
        self,
        *,
        knowledge_base_id: int | str | list[int | str],
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """在【指定知识库（支持单库或公私域多库联合）】内检索相似片段。

        ⚠️ knowledge_base_id 必传 —— 这不是可选参数，是安全边界。
        支持传入单个 ID（如 "3"）、逗号分隔的多个 ID（如 "3,6"）或 ID 列表（如 [3, 6]）。
        传空会抛 ValueError，而不是「检索全部」。

        Returns:
            [{chunkId, documentId, knowledgeBaseId, chunkIndex, content, score}, ...]
            score 是 0~1 的相似度（已把 Chroma 的 cosine 距离换算过来）
        """
        kb_ids = self._require_kb_ids(knowledge_base_id)
        k = top_k or settings.retrieval_top_k
        where_filter: dict[str, Any] = (
            {"knowledge_base_id": kb_ids[0]}
            if len(kb_ids) == 1
            else {"knowledge_base_id": {"$in": kb_ids}}
        )

        def _query() -> dict:
            return self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,     # ★ 单库精确匹配或多库 $in 联合检索
                include=["documents", "metadatas", "distances"],
            )

        raw = await asyncio.to_thread(_query)

        # Chroma 返回的是「批」的结构（外层多一层），取第 0 批
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        results = []
        for i, vec_id in enumerate(ids):
            meta = metas[i] or {}
            # cosine 距离 ∈ [0, 2] → 相似度 = 1 - 距离/2，落在 [0, 1]
            distance = dists[i] if i < len(dists) else None
            score = None if distance is None else round(1 - distance / 2, 6)

            results.append({
                "chunkId": vec_id,
                "documentId": meta.get("document_id"),
                "knowledgeBaseId": meta.get("knowledge_base_id"),
                "chunkIndex": meta.get("chunk_index"),
                "content": docs[i] if i < len(docs) else "",
                "score": score,
            })

        logger.info("检索完成：kb=%s，命中 %d 条", kb_ids, len(results))
        return results

    # ---------- 删除 ----------

    async def delete_by_document(self, document_id: int | str) -> None:
        """删除某个文档的所有向量。

        用途：① 重处理前清旧的（保证幂等）② 文档被删时清向量。
        """
        doc_id = int(document_id)

        def _delete() -> None:
            self._collection.delete(where={"document_id": doc_id})

        await asyncio.to_thread(_delete)

    async def count(self) -> int:
        """collection 里的总向量数（调试用）。"""
        return await asyncio.to_thread(self._collection.count)

    # ---------- 内部 ----------

    @staticmethod
    def _require_kb_id(value: int | str | None) -> int:
        """校验单个 knowledge_base_id。"""
        if value is None or str(value).strip() == "":
            raise ValueError(
                "knowledge_base_id 不能为空 —— "
                "缺少它会导致跨知识库检索，属于用户数据泄漏，已拒绝执行"
            )
        try:
            return int(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"knowledge_base_id 不是合法整数：{value!r}") from exc

    @classmethod
    def _require_kb_ids(cls, value: int | str | list[int | str] | None) -> list[int]:
        """校验并解析单或多个 knowledge_base_id（支持逗号分隔如 '3,6' 或列表 [3, 6]）。"""
        if value is None:
            raise ValueError(
                "knowledge_base_id 不能为空 —— "
                "缺少它会导致跨知识库检索，属于用户数据泄漏，已拒绝执行"
            )
        raw_items: list[int | str]
        if isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            text = str(value).strip()
            if not text:
                raise ValueError(
                    "knowledge_base_id 不能为空 —— "
                    "缺少它会导致跨知识库检索，属于用户数据泄漏，已拒绝执行"
                )
            raw_items = [part.strip() for part in text.split(",") if part.strip()]

        if not raw_items:
            raise ValueError(
                "knowledge_base_id 不能为空 —— "
                "缺少它会导致跨知识库检索，属于用户数据泄漏，已拒绝执行"
            )

        parsed = [cls._require_kb_id(item) for item in raw_items]
        return list(dict.fromkeys(parsed))


# 模块级单例
vector_store = VectorStore()
