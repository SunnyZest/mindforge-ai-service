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

logger = logging.getLogger(__name__)

# 单条请求的超时。嵌入比对话快，但批量大了也会慢
_TIMEOUT_SECONDS = 60.0

# DashScope 拒绝空字符串，用这个占位符顶替
_EMPTY_PLACEHOLDER = "（空）"


class EmbeddingService:
    """文本 → 向量。"""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            timeout=_TIMEOUT_SECONDS,
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

        # 空文本会让 DashScope 报错，先顶掉
        cleaned = [t if (t and t.strip()) else _EMPTY_PLACEHOLDER for t in texts]

        batch_size = max(1, settings.embedding_batch_size)
        vectors: list[list[float]] = []

        for start in range(0, len(cleaned), batch_size):
            batch = cleaned[start:start + batch_size]
            try:
                resp = await self._client.embeddings.create(
                    model=settings.embedding_model,
                    input=batch,
                )
            except Exception as exc:                    # noqa: BLE001
                raise RuntimeError(
                    f"嵌入失败（第 {start // batch_size + 1} 批，共 {len(batch)} 条）："
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            # 按 index 排序，保证顺序和输入一致（API 不保证返回顺序）
            ordered = sorted(resp.data, key=lambda d: d.index)
            vectors.extend([d.embedding for d in ordered])

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"嵌入结果数量不对：期望 {len(texts)} 条，实际 {len(vectors)} 条"
            )

        logger.debug("嵌入完成：%d 条，维度 %d", len(vectors), len(vectors[0]))
        return vectors

    async def embed_one(self, text: str) -> list[float]:
        """单条嵌入（检索时用）。"""
        result = await self.embed([text])
        return result[0]

    async def aclose(self) -> None:
        await self._client.close()


# 模块级单例
embedding_service = EmbeddingService()


async def _selftest() -> None:
    """手动验证：uv run python -m app.services.embedding"""
    vecs = await embedding_service.embed(["测试一下", "  ", "再测一条"])
    print(f"得到 {len(vecs)} 条向量，维度 {len(vecs[0])}")
    await embedding_service.aclose()


if __name__ == "__main__":
    asyncio.run(_selftest())
