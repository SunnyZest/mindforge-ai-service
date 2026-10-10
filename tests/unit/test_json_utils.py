"""json_utils 的纯函数测试。

这个模块是「模型输出 → 能入库的值」的边界。模型经常不按约定返回类型
（把年限写成 "3年"、把枚举写成「精通」），这里负责统一洗一遍。
所以边界行为要守住：洗错了会静默存进一个谁都认不出的值。
"""

import pytest

from app.services.json_utils import (
    JsonExtractError,
    clean_dict_list,
    clean_enum,
    clean_float,
    clean_int,
    clean_str,
    clean_str_list,
    extract_json_object,
)


class TestCleanStr:
    def test_none_and_blank_become_none(self):
        # 空串和 null 在库里语义不同，所以空值统一给 None
        assert clean_str(None) is None
        assert clean_str("") is None
        assert clean_str("   ") is None

    def test_strips_whitespace(self):
        assert clean_str("  hello  ") == "hello"

    def test_truncates_hard_without_ellipsis(self):
        # 硬切、不加省略号 —— 加了的话「…」会被当成正文存进库
        assert clean_str("abcdefghij", max_len=4) == "abcd"

    def test_no_max_len_keeps_everything(self):
        assert clean_str("abcdefghij") == "abcdefghij"

    def test_non_string_is_stringified(self):
        assert clean_str(123) == "123"


class TestCleanEnum:
    def test_case_insensitive(self):
        assert clean_enum("advanced", {"ADVANCED", "BEGINNER"}) == "ADVANCED"

    def test_unknown_value_becomes_none(self):
        # 「精通」不在白名单里 → None。
        # 宁可留空，也不存一个将来按 proficiency 排序时谁都认不出的值。
        assert clean_enum("精通", {"BEGINNER", "ADVANCED"}) is None

    def test_blank_becomes_none(self):
        assert clean_enum("  ", {"ADVANCED"}) is None


class TestCleanFloat:
    @pytest.mark.parametrize("raw,expected", [
        (3, 3.0),
        (3.5, 3.5),
        ("3", 3.0),
        ("3.5", 3.5),
        ("3年", 3.0),          # 模型很爱带单位
        ("约 5.5 年", 5.5),
        ("-2", None),          # 负数超出 lo 默认值 0
    ])
    def test_parses_various_forms(self, raw, expected):
        assert clean_float(raw) == expected

    def test_rounds_to_one_decimal(self):
        # 对齐 decimal(4,1) 的精度
        assert clean_float("3.567") == 3.6

    def test_out_of_range_becomes_none(self):
        assert clean_float(150, lo=0, hi=100) is None
        assert clean_float("200%", lo=0, hi=100) is None

    def test_bool_is_not_a_number(self):
        # bool 是 int 的子类，不特判的话 True 会变成 1.0 混进分数里
        assert clean_float(True) is None
        assert clean_float(False) is None

    def test_unparsable_becomes_none(self):
        assert clean_float("说不清") is None
        assert clean_float(None) is None


class TestCleanInt:
    def test_truncates_toward_zero(self):
        assert clean_int("3.9年") == 3

    def test_out_of_range(self):
        assert clean_int(200000, hi=100000) is None


class TestCleanStrList:
    def test_drops_blank_and_duplicates(self):
        assert clean_str_list(["a", "", "   ", "a", "b"]) == ["a", "b"]

    def test_respects_limit(self):
        assert clean_str_list(["a", "b", "c", "d"], limit=2) == ["a", "b"]

    def test_item_max_len(self):
        assert clean_str_list(["abcdef"], item_max_len=3) == ["abc"]

    @pytest.mark.parametrize("raw", [None, "a,b", 3, {"a": 1}])
    def test_non_list_returns_empty(self, raw):
        assert clean_str_list(raw) == []


class TestCleanDictList:
    def test_drops_non_dict_items(self):
        assert clean_dict_list([{"a": 1}, "x", 3, None, {"b": 2}]) == [{"a": 1}, {"b": 2}]

    def test_respects_limit(self):
        assert clean_dict_list([{"a": 1}, {"b": 2}, {"c": 3}], limit=2) == [{"a": 1}, {"b": 2}]

    def test_non_list_returns_empty(self):
        assert clean_dict_list("nope") == []


class TestExtractJsonObject:
    def test_plain_json(self):
        assert extract_json_object('{"name": "张伟"}') == {"name": "张伟"}

    def test_fenced_json(self):
        assert extract_json_object('```json\n{"name": "张伟"}\n```') == {"name": "张伟"}

    def test_fence_without_language_tag(self):
        assert extract_json_object('```\n{"a": 1}\n```') == {"a": 1}

    def test_surrounded_by_prose(self):
        text = '好的，这是提取结果：\n{"name": "张伟"}\n希望对你有帮助'
        assert extract_json_object(text) == {"name": "张伟"}

    def test_nested_braces_are_kept_whole(self):
        # 用 rfind 取最后一个 } 才能把最外层对象完整包住
        text = '前缀 {"outer": {"inner": 1}} 后缀'
        assert extract_json_object(text) == {"outer": {"inner": 1}}

    def test_empty_output(self):
        with pytest.raises(JsonExtractError, match="空内容"):
            extract_json_object("   ")

    def test_truncated_output_says_truncated(self):
        # ★ 截断和「格式跑偏」必须分开报：
        #   前者该去调大 extract_max_tokens，后者该去改提示词。
        #   混成一句话会让排查的人去翻提示词，方向全错。
        with pytest.raises(JsonExtractError) as excinfo:
            extract_json_object('{"name": "张伟", "skills": ["Java", "MySQL"')
        message = str(excinfo.value)
        assert "截断" in message
        assert "max_tokens" in message

    def test_no_json_at_all(self):
        with pytest.raises(JsonExtractError, match="找不到合法 JSON"):
            extract_json_object("模型今天不想干活，什么也没返回")

    def test_top_level_array_without_braces_is_rejected(self):
        # 本项目的结构化输出统一约定为对象（对象还能加字段，数组没有扩展余地）
        with pytest.raises(JsonExtractError):
            extract_json_object("[1, 2, 3]")

    def test_error_message_carries_raw_output(self):
        # 报错信息里必须带原始输出，否则「是模型没答还是格式跑偏」看不出来
        with pytest.raises(JsonExtractError) as excinfo:
            extract_json_object("这是一段没有 JSON 的模型输出，用于验证异常信息带原文")
        assert "这是一段没有 JSON 的模型输出" in str(excinfo.value)
