"""各 service 的 `_normalize_*` 输出契约。

这里守的是两条踩过坑才加上的不变量 —— 它们都表现为「接口通了但数据是空的」
或者「给了用户一个假的好消息」，光看日志发现不了：

① **输出键必须 camelCase**。写成 snake_case 时 Java 的 Jackson 对不上，
   字段静默变 null：HTTP 200、日志正常、数据是空的，排查方向全错。
   直接调 `_normalize_*` 就能在几毫秒内验证，不用起服务、不用调模型。

② **防虚构**。模型最常犯的错是把【岗位要求】当成【候选人的经历】抄进 evidence
   （要求「熟悉 Kafka」→ evidence 写「熟悉 Kafka」），那会让用户以为自己够格、
   投出去被刷。所以「没有依据不能说已匹配」是硬约束，得有用例守着。
"""

from typing import Any

import pytest

from app.services import interview, match, resume


def find_snake_case_keys(node: Any, path: str = "root") -> list[str]:
    """递归找出所有带下划线的键。"""
    offenders: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if "_" in key:
                offenders.append(f"{path}.{key}")
            offenders.extend(find_snake_case_keys(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            offenders.extend(find_snake_case_keys(value, f"{path}[{index}]"))
    return offenders


# ---------- ① camelCase 契约 ----------

MATCH_RAW: dict[str, Any] = {
    "overallScore": 65.4,
    "summary": "整体匹配一般：技能面基本对得上，缺 JVM 调优和带团队的经历。",
    "items": [
        {
            "category": "SKILL",
            "requirement": "熟悉 Kafka",
            "verdict": "MATCHED",
            "evidence": "2019-2021 用 Kafka 做过订单异步化",
            "advice": "保持，面试时把积压处理讲清楚",
        },
        {
            "category": "EXPERIENCE",
            "requirement": "有带团队经验",
            "verdict": "REAL_GAP",
            "evidence": None,
            "advice": "简历里没有带人经历，短期补不了",
        },
    ],
    "matchedSkills": ["Java", "MySQL", "Redis"],
    "missingSkills": ["JVM 调优"],
    "matchedExperience": ["3 年订单核心系统"],
    "missingExperience": ["团队管理"],
    "suggestions": ["补一个量化的性能提升案例"],
}

RESUME_RAW: dict[str, Any] = {
    "summary": "5 年 Java 后端，主要做订单与交易系统。",
    "name": "张伟",
    "phone": "13800000000",
    "email": "zhangwei@example.com",
    "location": "上海",
    "targetPosition": "Java 后端开发工程师",
    "yearsOfExperience": "5年",
    "education": [{"school": "某某大学", "degree": "本科", "major": "计算机科学与技术",
                   "startDate": "2015.09", "endDate": "2019.06"}],
    "workExperience": [{"company": "某某科技", "position": "Java 后端开发工程师",
                        "startDate": "2021.07", "endDate": "至今", "description": "订单中心"}],
    "projects": [{"name": "订单中台", "role": "后端负责人", "startDate": "2022.03",
                  "endDate": "2023.05", "description": "负责下单链路重构"}],
    "skills": [{"name": "Java", "level": "ADVANCED", "years": 5, "category": "LANGUAGE"}],
    "certificates": ["CET-6"],
    "selfEvaluation": "喜欢把复杂问题拆小。",
}

RESUME_ANALYSIS_RAW: dict[str, Any] = {
    "overallScore": 72.0,
    "summary": "经历本身够用，但表述缺少量化的结果。",
    "highlights": ["订单系统经验扎实"],
    "items": [
        {
            "target": "WORK_EXPERIENCE",
            "original": "负责订单中心开发",
            "issue": "看不出规模和结果",
            "suggestion": "补上日均单量、并发量、优化前后的指标",
            "rewritten": None,
            "needsFact": True,
            "missingFact": "日均单量和优化前后的 QPS/P99",
        }
    ],
}

INTERVIEW_QUESTIONS_RAW: dict[str, Any] = {
    "questions": [
        {
            "question": "讲讲 JVM 的内存模型和你做过的调优。",
            "questionType": "TECHNICAL",
            "expectedPoints": ["堆与栈", "GC 算法", "一次真实的调优案例"],
            "difficulty": "HARD",
        }
    ]
}

INTERVIEW_ANALYSIS_RAW: dict[str, Any] = {
    "score": 70.0,
    "technical": {"score": 65.0, "strengths": ["概念清楚"], "weaknesses": ["缺少真实案例"]},
    "communication": {"clarity": 80.0, "issues": ["有点绕"]},
    "missingPoints": ["GC 调优"],
    "comment": "方向对，但没有落到自己做过的具体事情上。",
    "followUp": "你那次 Full GC 的堆参数是怎么定的？",
}

INTERVIEW_REPORT_RAW: dict[str, Any] = {
    "overallScore": 9.9,
    "technicalScore": 7.7,
    "communicationScore": 18.0,
    "problemSolvingScore": 8.0,
    "strengths": ["表达有条理"],
    "weaknesses": ["缓存一致性没答上来"],
    "suggestions": ["复习缓存与数据库的一致性方案"],
    "reportText": "## 总体评价\n…",
}


@pytest.mark.parametrize("label,actual", [
    ("match._normalize", match._normalize(MATCH_RAW)),
    ("resume._normalize", resume._normalize(RESUME_RAW)),
    ("resume._normalize_analysis", resume._normalize_analysis(RESUME_ANALYSIS_RAW)),
    ("interview._normalize_questions",
     interview._normalize_questions(INTERVIEW_QUESTIONS_RAW, limit=5)),
    ("interview._normalize_analysis", interview._normalize_analysis(INTERVIEW_ANALYSIS_RAW)),
    ("interview._normalize_report", interview._normalize_report(INTERVIEW_REPORT_RAW)),
])
def test_output_keys_are_camel_case(label: str, actual: Any):
    offenders = find_snake_case_keys(actual)
    assert not offenders, (
        f"{label} 输出了 snake_case 的键：{offenders}。"
        f"Java 侧 Jackson 对不上会静默变 null（接口通了但数据是空的）。"
    )


# ---------- ② 防虚构 ----------

class TestMatchAntiHallucination:
    def test_matched_without_evidence_is_downgraded(self):
        """★ 核心不变量：没有依据就不能说「已匹配」。"""
        result = match._normalize({
            "items": [{"requirement": "熟悉 Kafka", "verdict": "MATCHED", "evidence": None}]
        })
        assert result["items"][0]["verdict"] == "UNCERTAIN"

    def test_matched_with_blank_evidence_is_downgraded(self):
        # evidence 是空串也要降级（clean_str 会把空串变成 None）
        result = match._normalize({
            "items": [{"requirement": "熟悉 Kafka", "verdict": "MATCHED", "evidence": "   "}]
        })
        assert result["items"][0]["verdict"] == "UNCERTAIN"

    def test_matched_with_evidence_stays_matched(self):
        """有依据的 MATCHED 不能被误降级 —— 否则五类结论里就没有「已匹配」了。"""
        result = match._normalize({
            "items": [{"requirement": "本科及以上", "verdict": "MATCHED",
                       "evidence": "2015.09-2019.06 某某大学 计算机科学与技术 本科"}]
        })
        assert result["items"][0]["verdict"] == "MATCHED"

    def test_non_matched_verdicts_keep_their_evidence_free_state(self):
        result = match._normalize({
            "items": [{"requirement": "精通 JVM 调优", "verdict": "REAL_GAP", "evidence": None}]
        })
        assert result["items"][0]["verdict"] == "REAL_GAP"

    def test_unknown_verdict_becomes_uncertain(self):
        result = match._normalize({
            "items": [{"requirement": "x", "verdict": "看起来还行", "evidence": "y"}]
        })
        assert result["items"][0]["verdict"] == "UNCERTAIN"

    def test_item_without_requirement_is_dropped(self):
        # 没有「要求」的条目，用户看不出在说哪一条
        result = match._normalize({
            "items": [{"verdict": "MATCHED", "evidence": "x"}, {"requirement": "有"}]
        })
        assert len(result["items"]) == 1
        assert result["items"][0]["requirement"] == "有"

    def test_items_are_capped(self):
        raw = {"items": [{"requirement": f"要求 {i}", "verdict": "UNCERTAIN"}
                         for i in range(30)]}
        assert len(match._normalize(raw)["items"]) <= 8


class TestResumeAnalyzeAntiHallucination:
    def test_needs_fact_forces_rewritten_to_none(self):
        """★ 模型「嘴上标了 needsFact、rewritten 里却已经编好了」时，必须把 rewritten 清掉。"""
        result = resume._normalize_analysis({
            "items": [{
                "target": "WORK_EXPERIENCE",
                "original": "负责订单中心开发",
                "suggestion": "补上量化指标",
                "rewritten": "负责订单中心开发，日均处理订单 500 万单，P99 从 800ms 降到 120ms",
                "needsFact": True,
                "missingFact": "日均单量和优化前后的 QPS/P99",
            }]
        })
        item = result["items"][0]
        assert item["needsFact"] is True
        assert item["rewritten"] is None, "标了 needsFact 就不能带改写结果 —— 那是编的"
        assert item["missingFact"] is not None

    def test_needs_fact_accepts_string_false(self):
        """★ 模型常把布尔写成字符串，`bool("false")` 是 True —— 不能直接用 bool()。"""
        result = resume._normalize_analysis({
            "items": [{"original": "x", "suggestion": "y",
                       "rewritten": "z", "needsFact": "false"}]
        })
        item = result["items"][0]
        assert item["needsFact"] is False
        assert item["rewritten"] == "z"
        assert item["missingFact"] is None

    def test_item_without_original_and_suggestion_is_dropped(self):
        result = resume._normalize_analysis({
            "items": [{"issue": "只有问题，没有原文也没有建议"}]
        })
        assert result["items"] == []

    def test_items_are_capped(self):
        raw = {"items": [{"original": f"原文 {i}", "suggestion": "建议"}
                         for i in range(30)]}
        assert len(resume._normalize_analysis(raw)["items"]) <= 5


class TestInterviewNormalize:
    def test_question_no_is_renumbered_from_one(self):
        """★ 题号由服务端重排，不信模型给的 —— 它经常从 0 开始、跳号或几道题共用一个号。
        Java 侧要按 questionNo 取「下一题」，编号一乱整个流程就断了。"""
        result = interview._normalize_questions({
            "questions": [
                {"question": "第一题", "questionNo": 0},
                {"question": "第二题", "questionNo": 0},
                {"question": "第三题", "questionNo": 99},
            ]
        }, limit=10)
        assert [q["questionNo"] for q in result] == [1, 2, 3]

    def test_question_without_text_is_dropped(self):
        result = interview._normalize_questions({
            "questions": [{"questionType": "TECHNICAL"}, {"question": "有题干"}]
        }, limit=10)
        assert len(result) == 1
        assert result[0]["questionNo"] == 1

    def test_questions_respect_limit(self):
        raw = {"questions": [{"question": f"第 {i} 题"} for i in range(50)]}
        assert len(interview._normalize_questions(raw, limit=3)) == 3

    def test_report_scores_are_clamped_to_100(self):
        result = interview._normalize_report({
            "overallScore": 150, "technicalScore": -5,
            "communicationScore": 80, "problemSolvingScore": 60,
        })
        assert result["overallScore"] is None
        assert result["technicalScore"] is None
        assert result["communicationScore"] == 80.0
        assert result["problemSolvingScore"] == 60.0
