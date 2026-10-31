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
import json
import logging
from pathlib import Path
from typing import Any

import chromadb

from app.config import settings
from app.services import ai_config

logger = logging.getLogger(__name__)

# app/services/vector_store.py → parents[2] 就是项目根
# 用绝对路径，避免「从哪个目录启动」影响向量库位置
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class VectorStore:
    """Chroma 向量库封装。

    chromadb 是同步 API，在 async 环境里直接调用会阻塞事件循环，
    所以所有操作都包了 asyncio.to_thread。

    ══════════════════════════════════════════════════════════════════
    ⚠️ 向量维度是【隐式】的，换 embedding 模型前必读
    ══════════════════════════════════════════════════════════════════

    MySQL 的 `document_chunk` 只存 `vector_id`（一个指针），**没有 embedding 列、
    也没有维度字段**；向量本身在 Chroma 的 HNSW 索引二进制里，SQL 查不到。

    Chroma 的 collection 在**第一次 add 时**由第一个向量的长度隐式锁定维度，
    之后写入不同长度的向量才会报错。

    所以「换模型」有两种后果，只有一种会被 Chroma 拦住：

    | 场景 | 结果 |
    | :--- | :--- |
    | 维度变了（1536 → 1024） | Chroma 写入报错，能发现 |
    | 维度没变（1536 → 另一个 1536 维模型） | **不报任何错**，但新旧向量混在同一 collection，检索静默变差 |

    第二种才是坑。所以这里在 `add()` / `query()` 前主动校验
    （见 `_check_embedding_compat`）：把首次写入时的**模型名 + 维度**登记到
    chroma 数据目录下的 `embedding_info.json`，之后每次比对，不一致直接抛异常
    并给出恢复步骤。

    ⚠️ 为什么不登记进 Chroma 的 collection metadata（试过，不行）：
    `collection.modify(metadata=...)` 会**整体覆盖**，而带上原来的
    `hnsw:space=cosine` 又会被 Chroma 拒绝（它认为你在改距离函数，直接抛
    "Changing the distance function of a collection once it is created is not supported"）。
    也就是说不带就丢 cosine、带就报错，两头堵。所以改成外挂文件 ——
    反正它就在 chroma 目录里，`rm -rf data/chroma` 会连它一起删掉，
    语义和「重建索引」是一致的。

    ⚠️ 模型名来自**数据库配置**（管理端「AI 模型配置」的 EMBEDDING 场景），
    不是 `.env` —— 也就是说管理员在界面上换个模型就能触发这个问题，
    所以校验只能放在运行时，启动时比对不了。
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

    # ---------- 维度校验 ----------

    def _embedding_info_path(self) -> Path:
        """维度登记文件 —— 和向量数据放一起，删数据目录时一起消失。"""
        return _PROJECT_ROOT / settings.chroma_path / "embedding_info.json"

    def _check_embedding_compat(self, dim: int) -> None:
        """校验本次向量与已登记的向量是同一套（同模型 + 同维度）。

        首次写入时把当前模型名和维度登记到 `embedding_info.json`，之后每次比对。

        ⚠️ 抛异常而不是只告警：维度/模型不一致时，写入会污染索引、检索会算错相似度，
        继续跑没有任何意义 —— 早失败早发现。调用方（pipeline）会捕获并把这个文档
        标记为 FAILED，不会变成 500。

        ⚠️ 恢复步骤写在异常消息里 —— 看到这个错的人多半不知道向量维度是隐式的，
        不写清楚他只会去翻 MySQL 的建表语句（那里什么都没有）。
        """
        path = self._embedding_info_path()
        current_model = ai_config.get(ai_config.SCENE_EMBEDDING).model

        if not path.exists():
            # 首次写入（或老库还没登记过）：把当前这套记下来。
            # 现有数据确实是这个模型算的，所以直接登记是安全的。
            info = {"embedding_model": current_model, "embedding_dim": dim}
            path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("向量库首次登记：model=%s dim=%d（写入 %s）",
                        current_model, dim, path)
            return

        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            # 文件坏了不该让整个服务不可用 —— 重建一份，但要说出来
            logger.warning("向量维度登记文件读不出来（%s），按当前配置重写：%s", path, e)
            info = {"embedding_model": current_model, "embedding_dim": dim}
            path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            return

        recorded_dim = info.get("embedding_dim")
        recorded_model = info.get("embedding_model")

        if recorded_dim != dim:
            raise RuntimeError(
                f"向量维度不一致：已登记 {recorded_dim} 维"
                f"（模型 {recorded_model or '未记录'}），本次算出 {dim} 维"
                f"（模型 {current_model}）。两种做法："
                f"① 改回原来的 embedding 模型；"
                f"② 确认要换模型的话，删掉 {settings.chroma_path} 下整个 chroma 数据目录"
                f"（含 embedding_info.json），重启后把所有文档重新向量化一遍"
                f"（MySQL 的 document_chunk 有全文，能重建）。"
                f"⚠️ 不要只删部分文档 —— 同一个 collection 里混两种维度的向量会检索出错。"
            )

        if recorded_model and current_model and recorded_model != current_model:
            raise RuntimeError(
                f"embedding 模型变了：已登记 {recorded_model}，当前配置是 {current_model}。"
                f"维度都是 {dim} 维，所以 Chroma 不会报错，"
                f"但两套语义空间算出的向量混在一起，检索结果会静默变差 —— 这是最隐蔽的一种。"
                f"两种做法：① 改回 {recorded_model}；"
                f"② 或删掉 {settings.chroma_path} 下整个 chroma 数据目录，重新向量化所有文档。"
            )

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

        # ★ 写入前校验维度 —— 不校验的话，换模型后新旧向量会混进同一个 collection
        await asyncio.to_thread(self._check_embedding_compat, len(embeddings[0]))

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

        # ★ 检索前也要校验 —— 只换模型不写新文档时，这里是唯一的拦截点。
        #   不校验的话，用新模型算的查询向量去比旧向量，结果会莫名其妙地差。
        await asyncio.to_thread(self._check_embedding_compat, len(query_embedding))
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
