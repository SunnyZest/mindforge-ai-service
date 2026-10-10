"""DashScope 对话模型封装。

和 embedding.py 一样，用 openai SDK 调 DashScope 的 OpenAI 兼容接口。
"""

import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# 对话生成比嵌入慢得多（要等模型逐 token 生成），超时给宽一点。
# ⚠️ 这是【客户端默认值】。抽取任务要吐几千 token，得单独传更长的超时
#    （见 settings.extract_timeout_seconds）—— 120 秒对详细简历是不够的。
_TIMEOUT_SECONDS = 120.0


# 没配 key 时用它占位 —— 理由同 embedding.py：
# 模块级单例在 import 时构造，直接传空 key 会让整个服务起不来。
_KEY_NOT_CONFIGURED = "NOT_CONFIGURED"


class LlmCallError(RuntimeError):
    """调用大模型失败（网络、鉴权、限流、返回空结果等）。

    单独定义而不是直接抛 RuntimeError，是为了让上层能**区分**这件事：
    调用模型失败是「上游依赖不可用」（该返回 502），
    而 RuntimeError 还可能是本服务自己的代码 bug（该返回 500）。
    两者混在一起的话，运维看到 500 会去查自己的代码，方向就错了。

    继承 RuntimeError 是兼容考虑 —— 已有的 `except RuntimeError` 照样能接住。
    """


class LlmService:
    """对话模型。"""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.dashscope_api_key or _KEY_NOT_CONFIGURED,
            base_url=settings.dashscope_base_url,
            timeout=_TIMEOUT_SECONDS,
            # ⚠️ 显式配，不吃 SDK 默认值（默认 2）——
            #    重试次数直接决定「最坏耗时 = 超时 × (1+重试)」，
            #    而 Java 侧的读超时是按这个最坏耗时定的（见 config.py 里的说明）。
            max_retries=settings.llm_max_retries,
        )

    @staticmethod
    def _ensure_configured() -> None:
        if not settings.dashscope_api_key:
            raise LlmCallError(
                "DASHSCOPE_API_KEY 未配置，无法调用对话模型。"
                "请在项目根目录的 .env 里填写（注意：本项目用 DashScope 的 "
                "OpenAI 兼容接口，配的是 DASHSCOPE_API_KEY，不是 OPENAI_API_KEY）"
            )

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """一次对话，返回完整回答。

        Args:
            messages: OpenAI 格式的消息列表 [{role, content}, ...]
            temperature: 温度。**问答场景应该调低** —— 默认 0.3，
                越低越不容易编造，但太死板。不要用默认的 1.0。
            max_tokens: 回答长度上限
            timeout: 本次调用的超时（秒）。不传用客户端默认的 120 秒。
                ⚠️ **抽取类任务必须传更长的值** —— 输出量大，生成时间远超对话。
                ⚠️ 而且它必须【小于】调用方（Java）的读超时，否则本服务
                报出来的错误根本没机会传回去，对方只会看到「连接断了」。

        Raises:
            LlmCallError: 调用失败
        """
        if not messages:
            raise ValueError("messages 不能为空")

        self._ensure_configured()

        try:
            resp = await self._client.chat.completions.create(
                model=settings.chat_model,
                messages=messages,
                temperature=settings.chat_temperature if temperature is None else temperature,
                max_tokens=max_tokens or settings.chat_max_tokens,
                timeout=timeout,
            )
        except Exception as exc:                     # noqa: BLE001
            raise LlmCallError(
                f"对话模型调用失败：{type(exc).__name__}: {exc}"
            ) from exc

        if not resp.choices:
            raise LlmCallError("对话模型返回了空结果")

        answer = (resp.choices[0].message.content or "").strip()

        if resp.usage:
            logger.info("对话完成：prompt=%d tokens, completion=%d tokens",
                        resp.usage.prompt_tokens or 0,
                        resp.usage.completion_tokens or 0)

        return answer

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        """流式对话 —— 逐段吐出增量文本。

        ⚠️ 和 chat() 的区别：这里只吐【增量文本】，不吐完整回答、不吐 usage。
        调用方要自己把增量拼起来（比如 chat.py 靠完整回答判断 hasAnswer）。

        ⚠️ 生成器里抛异常时，HTTP 响应头【已经发出去了】——
        调用方没法再改状态码，只能把错误当成流里的一个事件发出去。
        见 internal.py 的 /rag/chat/stream。

        Args:
            messages: OpenAI 格式的消息列表
            temperature: 温度，默认用 settings.chat_temperature
            max_tokens: 回答长度上限
            timeout: 本次调用的超时（秒）

        Yields:
            文本增量（可能为空串以外的任意片段，不保证按句/按词切）

        Raises:
            LlmCallError: 调用失败，或者流中途断了
        """
        if not messages:
            raise ValueError("messages 不能为空")

        self._ensure_configured()

        try:
            stream = await self._client.chat.completions.create(
                model=settings.chat_model,
                messages=messages,
                temperature=settings.chat_temperature if temperature is None else temperature,
                max_tokens=max_tokens or settings.chat_max_tokens,
                timeout=timeout,
                stream=True,
            )
        except Exception as exc:                     # noqa: BLE001
            # 连流都没建起来（鉴权 / 网络 / 限流）—— 这时还没吐过任何东西
            raise LlmCallError(
                f"对话模型调用失败：{type(exc).__name__}: {exc}"
            ) from exc

        try:
            async for chunk in stream:
                # 有些 chunk 只带 usage、没有 choices（末尾那一个），要跳过
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content
                if text:
                    yield text
        except Exception as exc:                     # noqa: BLE001
            # ⚠️ 中途断了 —— 前面已经吐出去的内容收不回来，
            #    调用方只能把这条错误追加到流末尾，让前端知道「后面没了」。
            raise LlmCallError(
                f"流式对话中断：{type(exc).__name__}: {exc}"
            ) from exc

    async def aclose(self) -> None:
        await self._client.close()


# 模块级单例
llm_service = LlmService()