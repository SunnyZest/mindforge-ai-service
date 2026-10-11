"""RAG 检索编排。

把「嵌入 query → 向量检索 → （可选）重排序 → 组装结果」串起来。

⚠️ 权限边界
本模块【不判权限】—— 调用方（Java）已经在调过来之前完成了权限检查，
并把确认过的 userId / knowledgeBaseId 传进来。这是文档 §十七 的硬性要求：
「Python 不自行决定用户是否有权限访问资源」。
本模块只负责「在给定的知识库范围内检索」。
"""

import logging

from app.services import ai_usage, rerank as rerank_service
from app.config import settings
from app.services.embedding import embedding_service
from app.services.rerank import RerankCallError
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)

# 开启重排时向量召回的放大倍数：召回宽一点（近似近邻的排序不够准），
# 交给 rerank 精排后收窄到 topK。倍数不宜太大 —— 候选多了 rerank 慢、且长尾没用。
RERANK_RECALL_MULTIPLIER = 3

# 放大后的召回上限，防止 topK 传很大时 rerank 候选失控
RERANK_RECALL_CAP = 30


async def search(
    *,
    knowledge_base_id: int | str,
    query: str,
    top_k: int | None = None,
    min_score: float | None = None,
    enable_rerank: bool = False,
) -> tuple[list[dict], bool]:
    """在指定知识库内做语义检索。

    Args:
        knowledge_base_id: 知识库 id —— **必传**，是数据隔离的安全边界
        query: 用户问题
        top_k: 返回条数上限，默认取配置
        min_score: 相似度下限（0~1），低于它的丢掉。None 表示不过滤
        enable_rerank: 是否用 rerank 模型精排。True 但未配置 RERANK 模型、
            或调用失败时，静默降级回向量序（reranked=False，不报错）

    Returns:
        (results, reranked) 二元组。
        results: [{chunkId, documentId, knowledgeBaseId, chunkIndex, content, score}, ...]
                 按相关性从高到低。开启重排时 score 是 rerank 的相关性分数（量纲和
                 向量相似度不同，但同为 0~1，前端展示无需区分）。
        reranked: 本次结果是否真的经过重排 —— 「请求了重排」不等于「重排生效了」，
                  响应里如实返回，Java 侧拿它决定要不要展示「已精排」标记。

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

    # 开重排时召回放大，多召回一些候选给 rerank 挑
    recall_k = k
    want_rerank = enable_rerank and rerank_service.configured()
    if want_rerank:
        recall_k = min(k * RERANK_RECALL_MULTIPLIER, RERANK_RECALL_CAP)

    query_vector = await embedding_service.embed_one(
        query.strip(), scene=ai_usage.SCENE_EMBEDDING)

    results = await vector_store.query(
        knowledge_base_id=knowledge_base_id,      # ★ 安全边界在这里生效
        query_embedding=query_vector,
        top_k=recall_k,
    )

    if min_score is not None:
        before = len(results)
        results = [r for r in results if (r.get("score") or 0) >= min_score]
        if len(results) < before:
            logger.debug("按 min_score=%.2f 过滤掉 %d 条低分结果",
                         min_score, before - len(results))

    reranked = False
    if want_rerank and results:
        try:
            ranked = await rerank_service.rerank(
                query.strip(), [r.get("content") or "" for r in results], top_n=k)
            # rerank 返回的下标对应当前的 results（min_score 过滤后的列表）
            results = [
                {**results[index], "score": score}
                for index, score in ranked
                if 0 <= index < len(results)
            ]
            reranked = True
        except RerankCallError as exc:
            # ★ 降级不报错：精排是增强，不是依赖。响应里 reranked=False 如实告知。
            logger.warning("重排序失败，退回向量序（reranked=False）：%s", exc)

    results = results[:k]

    logger.info("RAG 检索：kb=%s，topK=%d，命中 %d 条，最高分 %.4f%s",
                knowledge_base_id, k, len(results),
                results[0]["score"] if results and results[0]["score"] else 0,
                "（已重排）" if reranked else "")

    return results, reranked
