"""AI 使用量上报 —— 每次调完模型往 Java 写一条账。

【为什么是 Python 上报】
  只有拿到模型响应的一方才看得到 token 数。Java 只知道「我调了一次 Python」，
  不知道这次消耗了多少。

【三条可靠性约定】—— 这个功能最容易做错的地方就在这
  ① **尽力而为**：任何失败只打 WARNING。一次成功的 AI 调用不能因为「记不了账」
     而变成失败 —— 那是拿业务去换统计。
  ② **不 await**：用 create_task 发出去，不让每次 AI 调用多等一个 HTTP 往返。
     ⚠️ 但必须**持有 task 的强引用** —— asyncio 对 task 只持弱引用，
        不持有的话可能在跑完之前被 GC 掉（官方文档明确提过这个坑）。
  ③ **不重试、不落本地队列**：统计功能丢几条不影响决策；
     加重试或本地队列要引入状态、定时 flush、退出时 flush，代价远大于收益。
"""

import asyncio
import contextvars
import logging
from typing import Any

from app.clients.java_client import java_client

logger = logging.getLogger(__name__)

# ---------- 场景名 ----------
# 用常量而不是散落的字符串字面量：写错了不会报错，只会让统计分不出组
# （「RESUME_PARS」和「RESUME_PARSE」在 SQL 里是两个 scene）。
SCENE_RAG_CHAT = "RAG_CHAT"
SCENE_RESUME_PARSE = "RESUME_PARSE"
SCENE_RESUME_ANALYZE = "RESUME_ANALYZE"
SCENE_RESUME_MATCH = "RESUME_MATCH"
SCENE_JOB_PARSE = "JOB_PARSE"
SCENE_INTERVIEW_GENERATE = "INTERVIEW_GENERATE"
SCENE_INTERVIEW_ANALYZE = "INTERVIEW_ANALYZE"
SCENE_INTERVIEW_REPORT = "INTERVIEW_REPORT"
SCENE_EMBEDDING = "EMBEDDING"
SCENE_RERANK = "RERANK"

# 调用方没标场景时的兜底。
#
# ⚠️ 刻意「记一条 UNKNOWN」而不是「不记」—— 不记的话统计里看不到，
#    漏标就会一直是隐形的；记成 UNKNOWN 至少能在报表里发现「有些调用没分类」。
SCENE_UNKNOWN = "UNKNOWN"

SUCCESS = "SUCCESS"
FAILED = "FAILED"

# 当前请求的 traceId（Java 通过 X-Trace-Id 头传过来）。
#
# 用 contextvar 而不是全局变量：本服务是异步的，一个进程同时处理多个请求，
# 全局变量会串台（A 请求的记账挂到 B 的 traceId 上）。
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("mf_trace_id", default="")

# 当前请求的用户 id（Java 通过 X-User-Id 头传过来）。
#
# 为什么走 header 而不是给每个 service 函数加 user_id 参数：
#   parse_resume / match_resume / parse_job 这些函数本来就没有 user_id，
#   为记账去改 11 个函数的签名和它们的调用方，改动面远大于收益。
#   Java 那边本来就知道是谁触发的，多带一个 header 最省事。
_user_id: contextvars.ContextVar[str] = contextvars.ContextVar("mf_user_id", default="")

# 在飞的上报 task。只为了持有强引用 —— 见模块开头第 ② 条。
_pending: set[asyncio.Task] = set()


def set_trace_id(trace_id: str | None) -> None:
    """由 ASGI 中间件在每个请求开始时设置。"""
    _trace_id.set((trace_id or "").strip())


def set_user_id(user_id: str | None) -> None:
    """由 ASGI 中间件在每个请求开始时设置。"""
    _user_id.set((user_id or "").strip())


def current_trace_id() -> str:
    return _trace_id.get()


def usage_fields(usage: Any) -> tuple[int, int, int]:
    """从 SDK 的 usage 对象里取 (prompt, completion, total)。

    ⚠️ embedding 的响应只有 prompt_tokens / total_tokens，没有 completion_tokens；
       而且不同提供方可能少字段，所以一律 getattr + 兜底。
    """
    if usage is None:
        return 0, 0, 0
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or (prompt + completion)
    return prompt, completion, total


def record(
    *,
    scene: str,
    user_id: str | int | None = None,
    provider: str | None = None,
    model: str | None = None,
    usage: Any = None,
    duration_ms: int | None = None,
    status: str = SUCCESS,
    error_message: str | None = None,
) -> None:
    """记一条账。**同步函数**，内部起 task 发出去，不阻塞调用方。

    调用方（llm.py / embedding.py）拿到模型响应后直接调它就行，
    不用 await、也不用关心失败。
    """
    prompt, completion, total = usage_fields(usage)
    # 显式传的优先；没传就用中间件从 X-User-Id 里接住的
    resolved_user = user_id if user_id is not None else (_user_id.get() or None)
    payload = {
        "userId": None if resolved_user is None else str(resolved_user),
        "requestId": current_trace_id() or None,
        "scene": scene,
        "provider": provider,
        "modelName": model,
        "promptTokens": prompt,
        "completionTokens": completion,
        "totalTokens": total,
        "durationMs": duration_ms,
        "status": status,
        # 截断：Java 那边这一列是 varchar(1000)，超了整条记录会被丢掉
        "errorMessage": (error_message or None) and error_message[:1000],
    }

    try:
        task = asyncio.get_running_loop().create_task(_send(payload))
    except RuntimeError:
        # 不在事件循环里（同步脚本 / 单元测试）—— 直接跳过。
        # 记账失败不该让调用方出问题，静默跳过就好。
        logger.debug("不在事件循环里，跳过 AI 使用量上报 scene=%s", scene)
        return

    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def _send(payload: dict) -> None:
    try:
        await java_client.report_ai_usage(payload)
    except Exception as exc:                            # noqa: BLE001
        # ⚠️ 只打 WARNING，绝不往上抛。这是「尽力而为」的落点。
        logger.warning("上报 AI 使用量失败（只影响统计，不影响业务）：%s", exc)


async def drain(timeout: float = 3.0) -> None:
    """等所有在飞的上报发完。关服务时调一次，别把最后几条账丢了。"""
    if not _pending:
        return
    pending = list(_pending)
    try:
        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        logger.warning("还有 %d 条 AI 使用量没上报完，直接关闭", len(_pending))
