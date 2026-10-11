"""重排序（rerank）—— 对向量检索的候选做精排。

【为什么需要】
  向量检索的分数是「语义相似度」，对「和问题像」敏感，对「能回答问题」不敏感。
  rerank 模型是交叉编码器，把 query 和每条候选放在一起打分，
  排序质量明显高于双塔召回 —— 典型用法：向量召回宽一点（topK × 3），
  rerank 收窄到 topK。

【为什么单独一个模块、不走 OpenAI 兼容协议】
  DashScope 的 rerank API（gte-rerank 系列）不在 /compatible-mode/v1 的
  OpenAI 兼容层里，是原生 REST 端点，报文结构也不同 —— 复用不了 llm.py 的
  OpenAI client。base_url 按管理端配的来，剥掉兼容层路径后拼原生端点。

【可靠性约定：rerank 是增强，不是依赖】
  调用失败（超时 / 没配模型 / 返回畸形）只抛 RerankCallError，由 rag.search
  降级回向量序并打 WARNING —— 一次检索不能因为「精排失败」整个失败。
  ⚠️ 这和「记忆/记账失败只 WARNING」不同：rerank 失败要显式降级而不是静默吞掉，
  因为调用方（Java）会拿响应里的 reranked 字段决定要不要展示「已精排」标记。
"""

import logging

import httpx

from app.services import ai_usage
from app.services.ai_config import get as get_config

logger = logging.getLogger(__name__)

# 兼容层路径（OpenAI 协议的 base_url 常见后缀）。剥掉它才是 DashScope 根域名。
_COMPAT_SUFFIXES = ("/compatible-mode/v1", "/compatible-mode")

# 原生 rerank 端点（拼在根域名后）
_RERANK_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"

# 候选文本上限：DashScope rerank 对输入总量有限制，超长候选截断保平安
_MAX_DOC_CHARS = 4000


class RerankCallError(Exception):
    """rerank 调用失败（网络 / 认证 / 响应畸形）。调用方降级，不往外抛。"""


def configured() -> bool:
    """RERANK 模型配好了吗（管理端配置优先，环境变量不提供 rerank 兜底）。"""
    return bool(get_config(ai_usage.SCENE_RERANK).model.strip())


def _endpoint(base_url: str) -> str:
    root = base_url.strip().rstrip("/")
    for suffix in _COMPAT_SUFFIXES:
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    return root + _RERANK_PATH


async def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
    """调用 rerank 模型，返回 [(候选下标, 相关性分数)]，按分数从高到低。

    Raises:
        RerankCallError: 配置缺失 / 请求失败 / 响应畸形。
    """
    config = get_config(ai_usage.SCENE_RERANK)
    if not config.model.strip():
        raise RerankCallError("RERANK 模型未配置（管理端 scene=RERANK 没有启用的配置）")
    if not documents:
        return []

    payload = {
        "model": config.model,
        "input": {
            "query": query,
            "documents": [doc[:_MAX_DOC_CHARS] for doc in documents],
        },
        "parameters": {
            "return_documents": False,
            **({"top_n": top_n} if top_n is not None else {}),
        },
    }

    try:
        async with httpx.AsyncClient(timeout=config.timeout_seconds or 10) as client:
            response = await client.post(
                _endpoint(config.base_url),
                json=payload,
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPStatusError as exc:
        raise RerankCallError(
            f"rerank 接口返回 {exc.response.status_code}：{exc.response.text[:200]}") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise RerankCallError(f"rerank 请求失败：{exc}") from exc

    results = (body.get("output") or {}).get("results")
    if not isinstance(results, list):
        raise RerankCallError(f"rerank 响应缺 output.results：{str(body)[:200]}")

    ranked: list[tuple[int, float]] = []
    for item in results:
        try:
            ranked.append((int(item["index"]), float(item["relevance_score"])))
        except (KeyError, TypeError, ValueError):
            logger.warning("跳过一条畸形 rerank 结果：%s", item)

    # 上报使用量：rerank 也是真实的模型调用（尽力而为，失败不影响业务）
    usage = body.get("usage") or {}
    ai_usage.record(
        scene=ai_usage.SCENE_RERANK,
        provider=config.provider,
        model=config.model,
        usage=usage,
        status=ai_usage.SUCCESS,
    )

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    logger.info("rerank 完成：%d 条候选 → %d 条结果，最高分 %.4f",
                len(documents), len(ranked), ranked[0][1] if ranked else 0)
    return ranked
