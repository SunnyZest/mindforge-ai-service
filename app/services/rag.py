"""RAG 检索编排。

把「嵌入 query → 向量检索 → 组装结果」串起来。

⚠️ 权限边界
本模块【不判权限】—— 调用方（Java）已经在调过来之前完成了权限检查，
并把确认过的 userId / knowledgeBaseId 传进来。这是文档 §十七 的硬性要求：
「Python 不自行决定用户是否有权限访问资源」。
本模块只负责「在给定的知识库范围内检索」。
"""

import logging

from app.services import ai_usage
from app.config import settings
from app.services.embedding import embedding_service
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)


async def search(
    *,
    knowledge_base_id: int | str,
    query: str,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[dict]:
    """在指定知识库内做语义检索。

    Args:
        knowledge_base_id: 知识库 id —— **必传**，是数据隔离的安全边界
        query: 用户问题
        top_k: 返回条数上限，默认取配置
        min_score: 相似度下限（0~1），低于它的丢掉。None 表示不过滤

    Returns:
        [{chunkId, documentId, knowledgeBaseId, chunkIndex, content, score}, ...]
        按相似度从高到低

    Raises:
        ValueError: 参数不合法（query 为空 / knowledgeBaseId 缺失）
    """
    if not query or not query.strip():
        raise ValueError("查询内容不能为空")

    # knowledge_base_id 的校验在 vector_store 里（那里是最后一道闸），
    # 但这里也提前拦一下 —— 早失败早报错，不用白调一次嵌入接口
    if knowledge_base_id is None or str(knowledge_base_id).strip() == "":
        raise ValueError("knowledgeBaseId 不能为空")

    k = top_k or settings.retrieval_top_k
    # 上限兜底：topK 传太大会把一堆无关片段塞进 prompt，既费钱又降低答案质量
    k = min(max(1, k), settings.max_retrieval_top_k)

    query_vector = await embedding_service.embed_one(
        query.strip(), scene=ai_usage.SCENE_EMBEDDING)

    results = await vector_store.query(
        knowledge_base_id=knowledge_base_id,      # ★ 安全边界在这里生效
        query_embedding=query_vector,
        top_k=k,
    )

    if min_score is not None:
        before = len(results)
        results = [r for r in results if (r.get("score") or 0) >= min_score]
        if len(results) < before:
            logger.debug("按 min_score=%.2f 过滤掉 %d 条低分结果",
                         min_score, before - len(results))

    logger.info("RAG 检索：kb=%s，topK=%d，命中 %d 条，最高分 %.4f",
                knowledge_base_id, k, len(results),
                results[0]["score"] if results and results[0]["score"] else 0)

    return results
