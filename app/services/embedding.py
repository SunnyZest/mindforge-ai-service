"""DashScope 嵌入模型封装。

用 openai SDK 调 DashScope 的 OpenAI 兼容接口 —— 不需要 DashScope 专有 SDK。

注意和 java_client 的区别：
  java_client → 内网地址，trust_env=False（不走代理）
  这里        → 公网地址，走默认（有代理的话应该走代理）
"""

import asyncio
import logging

from openai import AsyncOpenAI

from app.config import settings
from app.services import ai_config

logger = logging.getLogger(__name__)

# 单条请求的超时。嵌入比对话快，但批量大了也会慢
_TIMEOUT_SECONDS = 60.0

# DashScope 拒绝空字符串，用这个占位符顶替
_EMPTY_PLACEHOLDER = "（空）"


# 没配 key 时用它占位，让 client 能构造出来。
#
# ⚠️ 为什么要这样：模块级单例是在 import 时构造的，如果这里直接传空 key，
# AsyncOpenAI 会抛 OpenAIError，导致【整个服务起不来】——
# 连带 /health 都用不了，而错误信息还是误导性的
# 「set the OPENAI_API_KEY environment variable」（本项目用的是 DashScope，
# 根本没配 OPENAI_API_KEY）。所以这里降级成「能启动、调用时才报错」，
# 并把错误信息说清楚。
_KEY_NOT_CONFIGURED = "NOT_CONFIGURED"


class EmbeddingCallError(RuntimeError):
    """调用嵌入模型失败（网络、鉴权、限流、返回数量不对等）。

    和 `llm.LlmCallError` 是同一个思路：把「上游依赖不可用」和「本服务代码 bug」分开。
    ⚠️ 不分开的话，向量化失败（key 过期 / 额度用完 / 网络不通）会报成 500，
    运维去查自己的代码 —— 方向全错。

    继承 RuntimeError 是兼容考虑：已有的 `except RuntimeError` 照样能接住。

    ⚠️ 这个异常会【穿透到检索】—— 因为 rag.search 第一步就是嵌入查询词。
    所以对话接口遇到它，表现是「连引用都拿不到」，而不是「引用有、答案没有」。
    """


class EmbeddingService:
    """文本 → 向量。"""

    def __init__(self) -> None:
        # ⚠️ 懒建 client —— 理由同 llm.py：api_key / base_url / 模型名可能来自
        #    管理端配置，而那份配置每 60 秒刷新一次（见 app/services/ai_config.py）。
        self._openai: AsyncOpenAI | None = None
        self._openai_key: tuple[str, str] | None = None

    def _client(self) -> AsyncOpenAI:
        """按【当前】配置拿 client；配置变了就重建。"""
        config = ai_config.get(ai_config.SCENE_EMBEDDING)
        key = (config.api_key, config.base_url)
        if self._openai is None or self._openai_key != key:
            self._openai = AsyncOpenAI(
                api_key=config.api_key or _KEY_NOT_CONFIGURED,
                base_url=config.base_url,
                timeout=_TIMEOUT_SECONDS,
            )
            self._openai_key = key
        return self._openai

    @staticmethod
    def _ensure_configured() -> None:
        """调用前检查 key —— 报错要说清是哪个配置项，别让人去猜。"""
        if not ai_config.get(ai_config.SCENE_EMBEDDING).api_key:
            raise EmbeddingCallError(
                "没有可用的嵌入模型 API Key。请检查两处："
                "① 管理端「AI 模型配置」里有没有配 EMBEDDING 场景；"
                "② 项目根目录 .env 里的 DASHSCOPE_API_KEY"
                "（本项目用 DashScope 的 OpenAI 兼容接口，配的是 DASHSCOPE_API_KEY，"
                "不是 OPENAI_API_KEY）"
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入，自动按 API 上限分批。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            和输入顺序、长度一一对应的向量列表

        Raises:
            RuntimeError: 任何一批失败
        """
        if not texts:
            return []

        self._ensure_configured()

        # 空文本会让 DashScope 报错，先顶掉
        cleaned = [t if (t and t.strip()) else _EMPTY_PLACEHOLDER for t in texts]

        batch_size = max(1, settings.embedding_batch_size)
        vectors: list[list[float]] = []

        for start in range(0, len(cleaned), batch_size):
            # 一次处理的条数(长度)
            batch = cleaned[start:start + batch_size]
            try:
                resp = await self._client().embeddings.create(
                    model=ai_config.get(ai_config.SCENE_EMBEDDING).model,
                    input=batch,
                )
            except Exception as exc:                    # noqa: BLE001
                raise EmbeddingCallError(
                    f"嵌入失败（第 {start // batch_size + 1} 批，共 {len(batch)} 条）："
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            # 按 index 排序，保证顺序和输入一致（API 不保证返回顺序）
            ordered = sorted(resp.data, key=lambda d: d.index)
            vectors.extend([d.embedding for d in ordered])

        if len(vectors) != len(texts):
            raise EmbeddingCallError(
                f"嵌入结果数量不对：期望 {len(texts)} 条，实际 {len(vectors)} 条"
            )

        logger.debug("嵌入完成：%d 条，维度 %d", len(vectors), len(vectors[0]))
        return vectors

    async def embed_one(self, text: str) -> list[float]:
        """单条嵌入（检索时用）。"""
        result = await self.embed([text])
        return result[0]

    async def aclose(self) -> None:
        # client 是懒建的，可能压根没建过
        if self._openai is not None:
            await self._openai.close()


# 模块级单例
embedding_service = EmbeddingService()


async def _selftest() -> None:
    """手动验证：uv run python -m app.services.embedding"""
    vecs = await embedding_service.embed(["测试一下", "  ", "再测一条"])
    print(f"得到 {len(vecs)} 条向量，维度 {len(vecs[0])}")
    await embedding_service.aclose()


if __name__ == "__main__":
    asyncio.run(_selftest())
