"""MindForge AI Service —— FastAPI 入口。

职责：
  · 提供 HTTP 服务（/health 等）
  · 启动时探测 Java 连通性（只打日志，不阻塞启动）
  · 启动轮询任务，主动去 Java 领待处理文档并跑流水线

启动：
    uv run uvicorn app.main:app --reload --port 8000
"""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import health, internal
from app.clients.java_client import java_client
from app.config import PROJECT_ROOT, settings
from app.services import ai_config, ai_usage, prompt_store
from app.services.embedding import EmbeddingCallError
from app.services.json_utils import JsonExtractError
from app.services.llm import LlmCallError
from app.services.parser import ParseError, UnsupportedFileTypeError
from app.worker.poller import poll_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)-5s - %(filename)s:%(lineno)d - %(message)s",

    datefmt="%Y-%m-%d %H:%M:%S",
)

# httpx 默认每次请求都打一条 INFO。
# 但轮询是每 5 秒一次的空转请求 —— 一天能刷出十几 MB 无意义日志，
# 真正有用的业务日志会被淹掉。调到 WARNING，只在出错时说话。
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def _probe_java() -> None:
    """启动时探测一次 Java —— 只打日志，不阻塞启动。

    用 create_task 而不是 await：Java 没起的时候不能把本服务卡死。
    """
    try:
        docs = await java_client.claim(limit=1)
        logger.info("✅ 已连上 Java（%s），当前待处理文档 %d 篇",
                    settings.java_base_url, len(docs))
        if docs:
            logger.info("   认领到：%s（会由轮询任务继续处理）",
                        [d.get("fileName") for d in docs])
    except Exception as exc:                       # noqa: BLE001
        logger.error("❌ 连不上 Java：%s", exc)
        logger.error("   请检查：① Java 是否已启动  ② JAVA_SERVICE_TOKEN 是否与 Java 一致")


async def _warm_up_remote_config() -> None:
    """启动时先拉一次 AI 模型配置和提示词，并打日志。

    目的是让「现在用的到底是哪套模型、哪套提示词」在启动横幅里一眼可见 ——
    「管理端改了配置没生效」这个问题，第一件事就是看这里。

    拉不到不算错（Java 可能还没起），后台协程会继续重试。
    """
    try:
        model_count = await ai_config.refresh_once()
    except Exception as exc:                            # noqa: BLE001
        logger.warning("   首次拉取 AI 模型配置失败（不影响启动）：%s", exc)
        model_count = 0

    if model_count:
        logger.info("   AI 模型配置 : %s（来自管理端）", ai_config.describe())
    else:
        logger.info("   AI 模型配置 : 管理端没配，全部回退 .env"
                    "（chat=%s, embedding=%s）",
                    settings.chat_model, settings.embedding_model)

    try:
        prompt_count = await prompt_store.refresh_once()
    except Exception as exc:                            # noqa: BLE001
        logger.warning("   首次拉取提示词失败（不影响启动）：%s", exc)
        prompt_count = 0

    if prompt_count:
        logger.info("   提示词配置   : %s（来自管理端）", prompt_store.describe())
    else:
        logger.info("   提示词配置   : 管理端没配，全部回退 prompts/*.txt")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("MindForge AI Service 启动中…")
    logger.info("   Java 后端    : %s", settings.java_base_url)
    logger.info("   轮询开关     : %s", settings.enable_poller)
    logger.info("   分块参数     : chunk_size=%d, overlap=%d",
                settings.chunk_size, settings.chunk_overlap)

    # 配置自检：把「配置漏了」在启动时就喊出来，别等到运行中才报 401
    if not settings.java_service_token:
        logger.error("JAVA_SERVICE_TOKEN 没配置！"
                     "所有 /internal 请求会被 Java 拒绝（401）。"
                     "请检查 %s 是否存在且已填。",
                     PROJECT_ROOT / ".env")
    if not settings.dashscope_api_key:
        # ⚠️ 只是警告不是错误：管理端可能配了 AI 模型，那就用管理端的。
        #    真正的「有没有 Key 可用」由 ai_config.get() 判断。
        logger.warning("DASHSCOPE_API_KEY 没配置 —— 如果管理端「AI 模型配置」里也没配，"
                       "文档向量化和问答都会失败。")

    # 启动时探测一次 Java（只打日志，不阻塞启动 —— Java 没起也不能把本服务卡死）
    probe_task = asyncio.create_task(_probe_java())

    # 远端配置（AI 模型 + 提示词）：先拉一次打日志，之后由后台协程每 60 秒刷新
    config_warmup_task = asyncio.create_task(_warm_up_remote_config())
    config_task = asyncio.create_task(ai_config.refresh_loop())
    prompt_task = asyncio.create_task(prompt_store.refresh_loop())

    # 轮询循环：本服务主动去 Java 领任务
    poller_task: asyncio.Task | None = None
    if settings.enable_poller:
        poller_task = asyncio.create_task(poll_loop())
    else:
        logger.warning("   轮询已关闭（ENABLE_POLLER=false）—— 不会处理任何文档")

    yield

    logger.info("MindForge AI Service 关闭中…")
    probe_task.cancel()
    config_warmup_task.cancel()
    config_task.cancel()
    prompt_task.cancel()
    if poller_task is not None:
        poller_task.cancel()
        # 等它真正退出，再关 httpx 连接池 —— 否则可能在半路被抽掉 client
        with suppress(asyncio.CancelledError):
            await poller_task
    for task in (probe_task, config_warmup_task, config_task, prompt_task):
        with suppress(asyncio.CancelledError):
            await task

    # ⚠️ 必须【在关连接池之前】把在飞的 AI 使用量上报发完 ——
    #    上报是 fire-and-forget 的，不等的话最后几条账会丢。
    await ai_usage.drain()

    await java_client.aclose()


app = FastAPI(
    title="MindForge AI Service",
    description="文档解析、分块、向量化、RAG 检索与对话 —— 只给 Java 后端调用",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(internal.router)


class RequestContextMiddleware:
    """把 Java 传过来的 traceId / userId 接住，供 AI 使用量记账使用。

    ⚠️ 用【纯 ASGI 中间件】而不是 {@code @app.middleware("http")}
       （后者基于 Starlette 的 BaseHTTPMiddleware）—— BaseHTTPMiddleware 会把下游
       放进另一个 task 里跑，contextvar 不一定传得下去。纯 ASGI 中间件是直接
       {@code await self.app(...)}，和路由处理函数在同一个 task 上下文里，
       这里 set 的 contextvar 处理函数一定看得到。

    ⚠️ 必须在【进入】下游之前 set —— 处理函数一开始跑就可能在调模型了。
    """

    def __init__(self, asgi_app) -> None:
        self._app = asgi_app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") == "http":
            headers = {k.lower(): v for k, v in (scope.get("headers") or [])}
            ai_usage.set_trace_id(_header(headers, b"x-trace-id"))
            ai_usage.set_user_id(_header(headers, b"x-user-id"))
        await self._app(scope, receive, send)


def _header(headers: dict[bytes, bytes], name: bytes) -> str:
    value = headers.get(name)
    return value.decode("latin-1") if value else ""


app.add_middleware(RequestContextMiddleware)


# ---------- 统一响应格式 ----------
#
# FastAPI 默认的错误格式是 {"detail": "..."}，和 Java 的 Result 不一样。
# Java 侧调用本服务时要解析响应，统一成 {code, message, data} 能省掉一套适配代码。

_CODE_BY_STATUS = {
    400: 40001,
    401: 40101,
    403: 40301,
    404: 40404,
    422: 40001,      # FastAPI 的参数校验失败
    500: 50000,
}


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": _CODE_BY_STATUS.get(exc.status_code, 50000),
            "message": exc.detail,
            "data": None,
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    """业务参数不合法 —— 比如查询为空、knowledgeBaseId 缺失。

    这类错误是调用方传错了参数，属于 4xx 而不是 5xx。
    """
    logger.warning("参数不合法：%s", exc)
    return JSONResponse(
        status_code=400,
        content={"code": 40001, "message": str(exc), "data": None},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI 的请求体校验失败。

    ⚠️ 这个必须单独注册：pydantic 的校验错误抛的是 RequestValidationError，
    不是 HTTPException，上面那个 handler 兜不住 —— 不注册的话会走 FastAPI 默认，
    返回 422 + {"detail": [...]}，格式和文档约定的 400 + {code, message, data} 对不上。

    这里统一成 400，和 Java 的 Result 格式一致。
    """
    # 取第一条错误拼成人话，别把整个 pydantic 错误数组丢给调用方
    first = (exc.errors() or [{}])[0]
    field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    detail = f"参数校验失败：{field} {first.get('msg', '')}".strip()

    logger.warning("请求体校验失败：%s", detail)
    return JSONResponse(
        status_code=400,
        content={"code": 40001, "message": detail, "data": None},
    )


@app.exception_handler(LlmCallError)
async def llm_call_error_handler(_: Request, exc: LlmCallError) -> JSONResponse:
    """调用大模型失败 → 502。

    ⚠️ 必须返回 502 而不是 500：这是【上游依赖不可用】，
    运维该去查模型服务的连通性 / 额度 / key，而不是查本服务的代码。
    混成 500 会把排查方向带偏。

    对齐 Java 的 ErrorCode：50202 = LLM 调用失败。
    """
    logger.error("调用大模型失败：%s", exc)
    return JSONResponse(
        status_code=502,
        content={"code": 50202, "message": str(exc), "data": None},
    )


@app.exception_handler(JsonExtractError)
async def json_extract_error_handler(_: Request, exc: JsonExtractError) -> JSONResponse:
    """模型没按约定输出 JSON → 502。

    ⚠️ 这个 handler 必须注册在 ValueError 之前被匹配到 ——
    Starlette 按异常类的 MRO 查找 handler，JsonExtractError 是 ValueError 的子类，
    所以它自己的 handler 会优先命中。注册顺序不影响，类型具体度决定。

    为什么不能让它落到 ValueError 那个 handler：那会返回 400「参数不合法」，
    但参数明明是对的 —— 是【模型输出跑偏了】。
    Java 侧看到 400 会以为是自己的请求有问题，方向又错了。
    """
    logger.error("模型输出不是合法 JSON：%s", exc)
    return JSONResponse(
        status_code=502,
        content={
            "code": 50201,
            "message": "模型没有按约定返回结构化数据，请重试或检查提示词",
            "data": None,
        },
    )


# ---------- 文件解析失败 ----------
#
# 这两类不注册的话会走 FastAPI 默认，返回 500 + {"detail": ...}，
# 格式和 {code, message, data} 对不上 —— Java 侧解析不了，只能报「AI 服务返回异常状态」。
#
# 都用 422 + 42201（Java 的 FILE_TYPE_UNSUPPORTED）：对用户来说
# 「这个 PDF 是扫描件、读不出文字」和「不支持这种文件」是同一件事 ——
# 都是「这个文件我们处理不了」。具体原因由 message 说清楚。

_FILE_UNREADABLE_CODE = 42201


@app.exception_handler(UnsupportedFileTypeError)
async def unsupported_file_type_handler(_: Request, exc: UnsupportedFileTypeError) -> JSONResponse:
    logger.warning("文件类型不支持：%s", exc)
    return JSONResponse(
        status_code=422,
        content={"code": _FILE_UNREADABLE_CODE, "message": str(exc), "data": None},
    )


@app.exception_handler(ParseError)
async def parse_error_handler(_: Request, exc: ParseError) -> JSONResponse:
    logger.warning("文件解析失败：%s", exc)
    return JSONResponse(
        status_code=422,
        content={"code": _FILE_UNREADABLE_CODE, "message": str(exc), "data": None},
    )


# ---------- 下载文件失败 ----------
#
# 不注册的话 httpx 的异常会走 FastAPI 默认 → 500 + {"detail": ...}，
# 格式和 {code, message, data} 对不上，Java 侧解析不了。


@app.exception_handler(httpx.HTTPStatusError)
async def http_status_error_handler(_: Request, exc: httpx.HTTPStatusError) -> JSONResponse:
    """对象存储返回了错误状态 → 422。

    最常见的原因是【预签名 URL 过期】—— Java 签发的下载地址有有效期，
    文件在队列里待久了就会失效。这不是本服务的错，也不是调用方参数写错，
    但对用户来说就是「文件读不到了」，所以给一句能指导行动的话（重试）。

    ⚠️ 不能只回「HTTP 403」：那样用户和排查的人都不知道该干什么。
    """
    logger.warning("下载文件失败：HTTP %s %s",
                   exc.response.status_code, exc.request.url)
    return JSONResponse(
        status_code=422,
        content={
            "code": _FILE_UNREADABLE_CODE,
            "message": f"文件下载失败（HTTP {exc.response.status_code}）——"
                       f"下载地址可能已过期，请重试",
            "data": None,
        },
    )


@app.exception_handler(httpx.RequestError)
async def http_request_error_handler(_: Request, exc: httpx.RequestError) -> JSONResponse:
    """连不上对象存储 → 502。

    和上面区分开：那个是「存储说不行」，这个是「根本连不上」——
    前者该重试或换地址，后者该去查网络 / 存储服务状态。
    """
    logger.error("连接对象存储失败：%s", exc)
    return JSONResponse(
        status_code=502,
        content={"code": 50201, "message": "连接文件存储失败，请稍后重试", "data": None},
    )


@app.exception_handler(EmbeddingCallError)
async def embedding_call_error_handler(_: Request, exc: EmbeddingCallError) -> JSONResponse:
    """调用嵌入模型失败 → 502。

    和 LlmCallError 一样是「上游依赖不可用」，所以也是 502 不是 500。

    ⚠️ 这个 handler 特别重要：嵌入是【检索的第一步】，
    它挂了的表现是「连引用片段都拿不到」—— 如果不注册，
    会走 FastAPI 默认的 500 + {"detail": ...}，Java 侧解析不了，
    最后报出来的是「AI 服务返回异常状态」，完全看不出是嵌入的问题。
    """
    logger.error("调用嵌入模型失败：%s", exc)
    return JSONResponse(
        status_code=502,
        content={
            "code": 50202,
            "message": f"向量化失败（检索用不了）：{exc}",
            "data": None,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """兜底：任何没被上面接住的异常 → 500 + 【统一信封】。

    ⚠️ 不注册这个的话，未预期的异常会走 FastAPI 默认，
    返回 `500 + {"detail": "Internal Server Error"}` —— 格式和
    `{code, message, data}` 对不上，Java 侧直接解析失败，
    最后用户看到的是「AI 服务返回异常状态」，根本不知道发生了什么。

    ⚠️ 只回一句人话，**不把异常内容返回给调用方** ——
    未预期异常的内容可能带内部路径、SQL、甚至配置片段。
    完整信息在服务日志里（logger.exception 会打堆栈）。
    """
    logger.exception("未处理的异常（这是本服务的 bug，不是调用方的问题）")
    return JSONResponse(
        status_code=500,
        content={"code": 50000, "message": "AI 服务内部错误", "data": None},
    )
