"""AI 模型配置 —— 优先用 Java 下发的，拿不到就回退环境变量。

【解决什么问题】
  管理端能维护 sys_ai_model（Java 加密存 Key），但真正调模型的是本服务。
  之前本服务只读自己的 .env —— 于是「管理端改了配置，没有任何效果」，
  配了等于没配。

【为什么是「后台预取 + 内存缓存」而不是按需拉】
  llm.py / embedding.py 的调用点都在同步上下文里（同理还有 prompt.render），
  把取配置改成 async 会让所有调用点都得加 await。所以：
    main.py 的 lifespan 起一个协程，每 60 秒拉一次、整体替换缓存；
    调用方只读内存，一行都不用改。
  代价是配置有最多 60 秒的生效延迟 —— 对「改模型配置」这种事完全可以接受。

【DB 是覆盖层，环境变量是默认值】
  sys_ai_model 里没配的 scene，就用 .env 里的值。所以：
    · 表是空的（现在的状态）→ 行为跟以前完全一样，不会因为这次改动出问题
    · 配了某一条 → 那个 scene 整条用 DB 的
    · 把配置删了 → 自动回退到 .env（缓存整体替换，不会留残影）
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.clients.java_client import java_client
from app.config import settings

logger = logging.getLogger(__name__)

SCENE_CHAT = "CHAT"
SCENE_EMBEDDING = "EMBEDDING"
SCENE_RERANK = "RERANK"

# 所有已定义的用途。snapshot() 必须把【每一个】都吐出来 ——
# 管理端要能看见「这个用途在用默认值」，只列配过的那几个等于把信息藏起来。
ALL_SCENES: tuple[str, ...] = (SCENE_CHAT, SCENE_EMBEDDING, SCENE_RERANK)

# 拉取间隔。配置变更最多延迟这么久生效 —— 改模型配置不是高频操作，
# 60 秒换来「调用方零改动」很划算。轮询循环同理，也用的这个量级。
REFRESH_SECONDS = 60

# 固定 +08:00，不用 ZoneInfo("Asia/Shanghai") ——
# Windows 没有系统时区库，用 ZoneInfo 要额外装 tzdata，漏装就是启动期崩。
# 本项目只面向国内，固定偏移与 Asia/Shanghai 完全等价。
_CST = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class ModelConfig:
    """一个用途（scene）下生效的模型配置。"""

    scene: str
    api_key: str
    base_url: str
    model: str
    # 提供方，只用于 AI 使用量统计（ai_usage_record.provider）
    provider: str = "dashscope"
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_seconds: int | None = None
    # "java" = 来自管理端配置；"env" = 回退到环境变量。
    # 只用于日志和排查 —— 「改了配置怎么没用」第一件事就是看这个。
    source: str = "env"


# scene -> 配置。只在 refresh_once 里整体替换，读的时候不用加锁
# （CPython 里 dict 的引用替换是原子的，读到旧的或新的都自洽）。
_cache: dict[str, ModelConfig] = {}
_last_error: str | None = None

# 上一次【成功】拉取的时间（北京时间）。
# ⚠️ 失败时保留上一次的值，不清空 —— 清空会让「配置多久没刷新了」这个信息丢失，
#    而排查「改了配置怎么没生效」时，第一眼要看的正是它。
_refreshed_at: datetime | None = None


def _env_config(scene: str) -> ModelConfig:
    """环境变量兜底配置。

    ⚠️ 模型名的默认值分场景：对话用 chat_model、向量化用 embedding_model。
       以前这两个是写死在 llm.py / embedding.py 里的，挪到这里统一管。
    """
    if scene == SCENE_CHAT:
        return ModelConfig(
            scene=scene,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            model=settings.chat_model,
            temperature=settings.chat_temperature,
            max_tokens=settings.chat_max_tokens,
            timeout_seconds=settings.extract_timeout_seconds,
            source="env",
        )
    if scene == SCENE_EMBEDDING:
        return ModelConfig(
            scene=scene,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            model=settings.embedding_model,
            source="env",
        )
    # RERANK 没有环境变量兜底 —— rerank 模型（gte-rerank 系列）和 chat 模型不是一回事，
    # 拿 chat_model 冒充只会让调用打到一个不存在的端点上。model 留空表示「未配置」，
    # rerank.configured() 据此判断，rag.search 未配置时直接跳过重排（走向量序）。
    return ModelConfig(
        scene=scene,
        api_key="",
        base_url=settings.dashscope_base_url,
        model="",
        source="env",
    )


def get(scene: str) -> ModelConfig:
    """取某个用途的配置。

    ⚠️ <b>永远不返回 None</b> —— 拿不到 Java 下发的就用环境变量。
       调用方不用写 `if config is None`，少一类空指针。
    """
    return _cache.get(scene) or _env_config(scene)


def describe() -> dict[str, str]:
    """当前生效的配置摘要（模型名 + 来源），用于启动日志和排查。

    ⚠️ 绝不含 api_key。
    """
    return {scene: f"{cfg.model}（{cfg.source}）" for scene, cfg in _cache.items()}


def last_error() -> str | None:
    """上一次拉取失败的原因（成功时清空）。"""
    return _last_error


def refreshed_at() -> datetime | None:
    """上一次【成功】拉取的时间；从没成功过就是 None。"""
    return _refreshed_at


@dataclass(frozen=True)
class SceneSnapshot:
    """一个用途当前真正在用的配置 —— 给管理端看的，不是给调用方用的。

    ⚠️ 只有 has_key 布尔，<b>绝不含 api_key 本身</b>。
       管理端要回答的问题是「这个用途到底通没通」，不是「Key 是什么」。
    """

    scene: str
    model: str
    source: str
    base_url: str
    has_key: bool


def snapshot() -> dict:
    """当前生效的配置快照，给管理端展示。

    ★ <b>为什么这个逻辑必须在 Python 侧</b>：
      生效规则是 `get(scene)` 的 `_cache.get(scene) or _env_config(scene)`，
      也就是「Java 下发的优先，没有就回退 .env」。这条规则只有本服务知道
      （.env 的模型名也只有本服务看得见）。Java 侧重算一遍必然漂移，
      而这种漂移最难查 —— 两边都说自己对。

    ★ 覆盖 {@link ALL_SCENES} 里的<b>每一个</b>用途，没配的也要出现
      （source="env"）：管理端需要知道「这个用途在用默认值」，
      只列配过的那些，等于把「没配」这件事藏起来了。

    ⚠️ 返回的 model 可能是空串（RERANK 没有环境变量兜底），
       这表示「未配置」，不是「配置成了空名字」。
    """
    scenes: list[dict] = []
    for scene in ALL_SCENES:
        cfg = get(scene)
        scenes.append({
            "scene": scene,
            "model": cfg.model,
            "source": cfg.source,
            # 对外是 camelCase，和 Java / 前端的命名对齐
            "baseUrl": cfg.base_url,
            # ⚠️ 只回布尔。api_key 一个字符都不能出现在响应里。
            "hasKey": bool(cfg.api_key.strip()),
        })

    return {
        "scenes": scenes,
        # ISO 带时区（+08:00），前端直接显示。从没成功拉过就是 None，
        # 前端据此显示「尚未同步」而不是一个假的时间。
        "refreshedAt": _refreshed_at.isoformat() if _refreshed_at else None,
        "lastError": _last_error,
    }


def _from_java(raw: dict) -> ModelConfig | None:
    """把 Java 下发的一条转成 ModelConfig；不完整就返回 None（跳过）。"""
    scene = (raw.get("scene") or "").strip().upper()
    model = (raw.get("modelName") or "").strip()
    if not scene or not model:
        # 没有 scene 或模型名，这条配置没有任何意义
        logger.warning("跳过一条不完整的模型配置：%s", {k: v for k, v in raw.items() if k != "apiKey"})
        return None

    return ModelConfig(
        scene=scene,
        # ⚠️ Key 可能是 null —— 比如本地模型本来就不需要 Key。
        #    这时用空串，调用前 _ensure_configured 会拦下来并说清原因。
        api_key=(raw.get("apiKey") or "").strip(),
        base_url=(raw.get("baseUrl") or settings.dashscope_base_url).strip(),
        model=model,
        provider=(raw.get("provider") or "dashscope").strip(),
        temperature=raw.get("temperature"),
        max_tokens=raw.get("maxTokens"),
        timeout_seconds=raw.get("timeoutSeconds"),
        source="java",
    )


async def refresh_once() -> int:
    """拉一次并整体替换缓存，返回生效的条数。

    ⚠️ 拉取失败时<b>保留上一次的缓存</b>，不清空 ——
       否则 Java 抖一下，所有 scene 就一起掉回环境变量，
       线上表现为「模型突然换了」，比「配置晚 60 秒生效」严重得多。
    """
    global _cache, _last_error, _refreshed_at

    try:
        models = await java_client.list_ai_models()
    except Exception as exc:                            # noqa: BLE001
        _last_error = str(exc)
        logger.warning("拉取 AI 配置失败，继续沿用上一次的（共 %d 条）：%s", len(_cache), exc)
        return 0

    fresh: dict[str, ModelConfig] = {}
    for raw in models:
        config = _from_java(raw)
        if config is None:
            continue
        # Java 侧已经按「默认优先」排好序了，所以同一个 scene 第一条胜出
        if config.scene in fresh:
            continue
        fresh[config.scene] = config

    _last_error = None
    # ★ 只在【成功】时更新。失败时保留上一次的时间，
    #   否则「配置已经 3 小时没刷新成功」这个关键信息就被抹掉了。
    #   抹掉微秒：这是个给人看的「上次同步时间」，微秒只是噪音。
    _refreshed_at = datetime.now(_CST).replace(microsecond=0)
    changed = fresh != _cache
    _cache = fresh

    if changed:
        # ⚠️ 只打 scene / 模型名 / 来源，绝不打 api_key
        logger.info("AI 配置已更新：%s", describe() or "（无，全部回退环境变量）")
    return len(fresh)


async def refresh_loop() -> None:
    """后台协程：每 REFRESH_SECONDS 拉一次，直到被取消。"""
    while True:
        try:
            await refresh_once()
        except asyncio.CancelledError:
            raise
        except Exception:                               # noqa: BLE001
            # refresh_once 自己已经兜住了 Java 的异常，走到这里说明是本模块的 bug。
            # 不能让它把协程打死 —— 配置刷新停了会很难发现。
            logger.exception("AI 配置刷新出错，%d 秒后重试", REFRESH_SECONDS)
        await asyncio.sleep(REFRESH_SECONDS)
