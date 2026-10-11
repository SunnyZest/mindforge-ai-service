"""提示词缓存 —— 优先用 Java 下发的，拿不到就回退 prompts/*.txt。

【为什么要这样】
  提示词正文在 Java（管理端维护，带版本管理和发布/回滚），但真正渲染它的是本服务。
  之前本服务只读 prompts/*.txt —— 于是「管理端改了提示词没有任何效果」。

【为什么是「后台预取 + 内存缓存」】
  和 ai_config 同一个模式：prompt.render() 是**同步函数**，调用点有 8 个，
  改成 async 会让所有调用点都得加 await。所以后台协程定时拉、内存缓存，
  调用方只读内存，一行都不用改。代价是最多 60 秒的生效延迟。

【DB 是覆盖层，文件是默认值】
  某个场景在 sys_prompt 里一条已发布的都没有 → 用 prompts/{scene}.txt。
  所以：表是空的（现在的状态）→ 行为跟以前完全一样；
       配了 → 那个场景用 DB 的；把配置删了 → 自动回退文件。
"""

import asyncio
import logging
from dataclasses import dataclass

from app.clients.java_client import java_client

logger = logging.getLogger(__name__)

# 拉取间隔。和 ai_config 保持一致 —— 改提示词不是高频操作，
# 60 秒换来「调用方零改动」很划算。
REFRESH_SECONDS = 60


@dataclass(frozen=True)
class PromptEntry:
    """一条已发布的提示词。"""

    id: str
    scene: str
    name: str
    version: int
    label: str | None
    is_default: bool
    user_selectable: bool
    content: str


# id -> 条目
_by_id: dict[str, PromptEntry] = {}
# scene -> 该场景下所有已发布的条目（默认的排最前）
_by_scene: dict[str, list[PromptEntry]] = {}
_last_error: str | None = None


def _from_java(raw: dict) -> PromptEntry | None:
    scene = (raw.get("scene") or "").strip().upper()
    content = raw.get("content")
    if not scene or not content:
        # 没有场景或没有正文的条目没有任何意义
        logger.warning("跳过一条不完整的提示词：scene=%s id=%s", scene, raw.get("id"))
        return None
    return PromptEntry(
        id=str(raw.get("id") or ""),
        scene=scene,
        name=(raw.get("name") or "").strip(),
        version=int(raw.get("version") or 1),
        label=(raw.get("label") or None),
        is_default=bool(raw.get("isDefault")),
        user_selectable=bool(raw.get("userSelectable")),
        content=content,
    )


def find(scene: str, prompt_id: str | None = None) -> PromptEntry | None:
    """按场景（+ 可选的指定 id）找一条已发布的提示词。

    找不到就返回 None —— 调用方回退到 prompts/{scene}.txt。

    ★ 安全边界：**指定 id 时只在「允许用户切换」的条目里找**。
      因为 Java 的 /internal/prompts 会把【所有已发布】的都下发下来
      （包括管理员标记为「不让用户切」的内部提示词，Python 得能按场景默认用它们），
      所以这里必须自己再挡一道 —— 否则用户可以传一个内部提示词的 id，
      把「简历解析必须吐特定 JSON」那套 prompt 换掉，直接让功能挂掉。
    """
    if not scene:
        return None

    scene_key = scene.strip().upper()

    if prompt_id:
        entry = _by_id.get(str(prompt_id))
        if entry is None:
            logger.warning("指定的提示词 id=%s 不存在（可能已下线），回退默认风格", prompt_id)
        elif entry.scene != scene_key:
            logger.warning("提示词 id=%s 属于场景 %s，和请求的 %s 不符，回退默认风格",
                           prompt_id, entry.scene, scene_key)
        elif not entry.user_selectable:
            # ★ 用户不能切到没开放给他的提示词
            logger.warning("提示词 id=%s（%s）没开放给用户切换，拒绝并回退默认风格",
                           prompt_id, entry.name)
        else:
            return entry

    entries = _by_scene.get(scene_key)
    if not entries:
        return None
    for entry in entries:
        if entry.is_default:
            return entry
    # 一个场景下没有标默认的：取第一条（Java 侧已按「默认优先」排序）
    return entries[0]


def describe() -> dict[str, str]:
    """当前生效的提示词摘要（场景 -> 变体名 v版本），用于启动日志和排查。"""
    return {
        scene: "、".join(f"{e.name} v{e.version}" + ("（默认）" if e.is_default else "")
                          for e in entries)
        for scene, entries in _by_scene.items()
    }


def last_error() -> str | None:
    return _last_error


async def refresh_once() -> int:
    """拉一次并整体替换缓存，返回生效的条数。

    ⚠️ 拉取失败时<b>保留上一次的缓存</b>，不清空 ——
       否则 Java 抖一下，所有提示词就一起掉回文件版，
       表现为「AI 的回答风格突然变了」，比「晚 60 秒生效」严重得多。
    """
    global _by_id, _by_scene, _last_error

    try:
        prompts = await java_client.list_prompts()
    except Exception as exc:                            # noqa: BLE001
        _last_error = str(exc)
        logger.warning("拉取提示词失败，继续沿用上一次的（共 %d 条）：%s",
                       len(_by_id), exc)
        return 0

    by_id: dict[str, PromptEntry] = {}
    by_scene: dict[str, list[PromptEntry]] = {}
    for raw in prompts:
        entry = _from_java(raw)
        if entry is None:
            continue
        by_id[entry.id] = entry
        by_scene.setdefault(entry.scene, []).append(entry)

    _last_error = None
    changed = by_scene != _by_scene
    _by_id = by_id
    _by_scene = by_scene

    if changed:
        logger.info("提示词已更新：%s", describe() or "（无，全部回退 prompts/*.txt）")
    return len(by_id)


async def refresh_loop() -> None:
    """后台协程：每 REFRESH_SECONDS 拉一次，直到被取消。"""
    while True:
        try:
            await refresh_once()
        except asyncio.CancelledError:
            raise
        except Exception:                               # noqa: BLE001
            logger.exception("提示词刷新出错，%d 秒后重试", REFRESH_SECONDS)
        await asyncio.sleep(REFRESH_SECONDS)
