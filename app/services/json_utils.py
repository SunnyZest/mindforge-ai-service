"""从模型输出里抠出 JSON —— 所有「让模型吐结构化数据」的场景共用。

为什么需要单独一个模块：**模型不会老老实实只输出 JSON**。实测常见三种情况：

  ① 包在 Markdown 代码块里：  ```json\n{...}\n```
  ② 前后带一句解释：          「好的，这是提取结果：\n{...}\n希望对你有帮助」
  ③ 代码块 + 解释同时出现

直接 json.loads 会在这三种情况下全部失败，而报错信息（"Expecting value: line 1
column 1"）完全看不出是格式问题还是模型没提取到内容 —— 排查起来很浪费时间。
所以统一在这里兜住，并且把「原始输出」带进异常信息里。
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 匹配开头的 ```json / ```JSON / ```
_FENCE_START = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_END = re.compile(r"\s*```$")
_FENCE_BLOCK = re.compile(r"```(?:json|JSON)?\s*(\{[\s\S]*?\})\s*```")


class JsonExtractError(ValueError):
    """模型输出里找不到合法的 JSON 对象。"""


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出里提取一个 JSON 对象。

    Raises:
        JsonExtractError: 三种情况 —— 空输出、不是合法 JSON、顶层不是对象。
            异常信息里带原始输出的片段，方便定位是「模型没答」还是「格式跑偏」。
    """
    if not text or not text.strip():
        raise JsonExtractError("模型返回了空内容")

    cleaned = _strip_fence(text.strip())

    # 先直接试 —— 大部分情况到这就成了
    obj = _try_load(cleaned)
    if obj is not None:
        return obj

    # 再试「截取最外层花括号」—— 对付前后带解释的情况。
    # 若寒暄语句里自带了 {}（如 "针对 foo() {} 的分析："），从左向右逐个尝试 '{' 起点。
    end = cleaned.rfind("}")
    if end != -1:
        pos = cleaned.find("{")
        while pos != -1 and pos < end:
            obj = _try_load(cleaned[pos:end + 1])
            if obj is not None:
                return obj
            pos = cleaned.find("{", pos + 1)

    # ★ 花括号不闭合 = 输出被 max_tokens 截断了。
    #   必须和「模型格式跑偏」分开报：前者该去调大 extract_max_tokens，
    #   后者该去改提示词。混成一句话会让排查的人去翻提示词，方向全错。
    opened, closed = cleaned.count("{"), cleaned.count("}")
    if opened > closed:
        raise JsonExtractError(
            f"模型输出被截断了（{opened} 个 '{{' 只闭合了 {closed} 个）——"
            f"多半是 max_tokens 不够，请调大 extract_max_tokens。"
            f"原始输出前 200 字：{text.strip()[:200]}"
        )

    raise JsonExtractError(
        f"模型输出里找不到合法 JSON。原始输出前 200 字：{text.strip()[:200]}"
    )


def _strip_fence(text: str) -> str:
    if text.startswith("```"):
        return _FENCE_END.sub("", _FENCE_START.sub("", text)).strip()
    m = _FENCE_BLOCK.search(text)
    if m:
        return m.group(1).strip()
    return text


def _try_load(text: str) -> dict[str, Any] | None:
    """能解析成 dict 就返回，否则 None（不抛异常，交给调用方继续尝试别的策略）。"""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        return obj
    # 顶层是数组或其他类型 → 也算失败。本项目的结构化输出统一约定为对象，
    # 这样将来加字段不会破坏调用方（数组则没有扩展余地）。
    logger.warning("模型输出了合法 JSON，但顶层不是对象：%s", type(obj).__name__)
    return None


# ---------- 字段清洗 ----------
#
# 下面这几个是给「把模型输出写进数据库」用的。
# 模型经常不按约定返回类型 —— 比如把熟练程度写成「精通」而不是 ADVANCED，
# 把年限写成 "3年" 而不是 3。直接入库会撞列类型（decimal 收字符串会报错），
# 或者存进一堆谁也认不出的值。所以在边界上统一转一道。


def clean_str(value: Any, *, max_len: int | None = None) -> str | None:
    """转成干净的字符串；空值返回 None（而不是空串 —— 空串在库里和 null 语义不同）。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:max_len] if max_len else text


def clean_enum(value: Any, allowed: set[str]) -> str | None:
    """只接受白名单里的值（大小写不敏感），其余返回 None。

    为什么返回 None 而不是原样保留：库里那些字段是有语义的枚举，
    存一个 "精通" 进去，将来按 proficiency 排序或筛选就全乱了。
    宁可留空，也不存一个谁都认不出的值。
    """
    text = clean_str(value)
    if text is None:
        return None
    upper = text.upper()
    return upper if upper in allowed else None


def clean_float(value: Any, *, lo: float = 0.0, hi: float = 99.9) -> float | None:
    """转成浮点数，超出范围或无法解析都返回 None。

    能处理 "3"、"3.5"、3、"3年" 以及 "2021-2024 (3年)" 这几种写法。
    hi 默认 99.9 是为了对齐 decimal(4,1) 的取值上限。
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if lo <= number <= hi:
            return round(number, 1)
        return None

    for match in re.finditer(r"-?\d+(?:\.\d+)?", str(value)):
        number = float(match.group())
        if lo <= number <= hi:
            return round(number, 1)
    return None


def clean_int(value: Any, *, lo: int = 0, hi: int = 100000) -> int | None:
    """转成整数，超出范围或无法解析返回 None。"""
    number = clean_float(value, lo=lo, hi=hi)
    return None if number is None else int(number)


def clean_bool(value: Any, *, default: bool = False) -> bool:
    """转成布尔值，兼容字符串 "true"/"false"/"是"/"否" 等模型输出，防止 bool("false") == True。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "y", "是", "必选", "必须"}:
            return True
        if s in {"false", "0", "no", "n", "否", ""}:
            return False
    return default


def clean_str_list(value: Any, *, item_max_len: int = 255, limit: int = 50) -> list[str]:
    """转成字符串列表，去掉空项和重复项，并限制条数。"""
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in value:
        text = clean_str(item, max_len=item_max_len)
        if text is None or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def clean_dict_list(value: Any, *, limit: int = 50) -> list[dict[str, Any]]:
    """转成「对象列表」，丢掉不是对象的项。"""
    if not isinstance(value, list):
        return []
    return [item for item in value[:limit] if isinstance(item, dict)]
