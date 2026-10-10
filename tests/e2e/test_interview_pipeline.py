"""真实链路：AI 面试五步 创建 → 出题 → 开始 → 逐题作答 → 结束。

Java 侧的 InterviewApiTest 用 mock 覆盖了状态机；这里验证 mock 测不出来的：
模型出的题和简历/岗位对得上吗？逐题评分有没有真的看参考答案要点？
报告里的综合分和逐题分对得上吗？
"""

import pytest

from tests.e2e.helpers import SAMPLE_JD, SAMPLE_RESUME, assert_camel_case, ok

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def interview(api, auth, cleanup):
    """造一场带简历和岗位的面试，并出好题。

    module 级：出题是一次真模型调用，整个模块共用一场，别每题都重出一遍。
    """
    resume = ok(api.post(
        "/app/resumes",
        headers=auth,
        files={"file": ("e2e-面试用简历.txt", SAMPLE_RESUME.encode("utf-8"), "text/plain")},
    ))
    cleanup.append((f"/app/resumes/{resume['id']}", f"简历 {resume['id']}"))
    ok(api.post(f"/app/resumes/{resume['id']}/parse", headers=auth))

    job = ok(api.post("/app/jobs", headers=auth, json={
        "title": "高级 Java 后端开发工程师",
        "companyName": "某某互联网公司",
        "jobDescription": SAMPLE_JD,
    }))
    cleanup.append((f"/app/jobs/{job['id']}", f"职位 {job['id']}"))
    ok(api.post(f"/app/jobs/{job['id']}/parse", headers=auth))

    created = ok(api.post("/app/interviews", headers=auth, json={
        "title": "e2e 模拟面试",
        "resumeId": resume["id"],
        "jobId": job["id"],
        "interviewType": "TECHNICAL",
        "difficulty": "MEDIUM",
        "questionCount": 3,
    }))
    interview_id = created["id"]
    cleanup.append((f"/app/interviews/{interview_id}", f"面试 {interview_id}"))

    detail = ok(api.post(f"/app/interviews/{interview_id}/generate", headers=auth))
    return {"id": interview_id, "questions": detail.get("questions") or []}


def test_generate_produces_questions(interview):
    questions = interview["questions"]
    assert questions, "出题应该产出题目"
    assert [q["questionNo"] for q in questions] == list(range(1, len(questions) + 1)), (
        "题号必须由 1 开始连续 —— Java 侧靠它取「下一题」"
    )
    for question in questions:
        assert_camel_case(question)
        assert question.get("question"), f"题目不能为空：{question}"
        # 进行中不能泄露参考答案要点
        assert not question.get("expectedPoints"), (
            f"面试还没开始就返回了参考答案要点，等于提前给答案：{question}"
        )


def test_expected_points_are_stored_server_side(interview, api, auth):
    """要点藏在服务端，结束后才放出来。"""
    ok(api.post(f"/app/interviews/{interview['id']}/start", headers=auth))
    detail = ok(api.get(f"/app/interviews/{interview['id']}", headers=auth))
    for question in detail.get("questions") or []:
        assert not question.get("expectedPoints"), (
            f"进行中不该返回 expectedPoints：{question}"
        )


def test_answer_scores_each_question(api, auth, interview):
    """逐题作答：好答案应该拿到分，且给出针对性的分析。"""
    questions = interview["questions"]
    assert len(questions) >= 2, "这场面试至少要有 2 道题才能验证逐题作答"

    good_answer = (
        "堆是线程共享的，栈是线程私有的。GC 主要针对堆，按分代回收："
        "新生代用复制算法（Eden + 两个 Survivor），老年代用标记整理。"
        "我们线上遇到过一次 Full GC 频繁，用 jstat 看到老年代增长快，"
        "最后定位到是批量任务把大对象直接放进了老年代，改成流式处理后 P99 从 900ms 降到 150ms。"
    )
    first = ok(api.post(f"/app/interviews/{interview['id']}/answers", headers=auth, json={
        "questionId": questions[0]["id"],
        "answer": good_answer,
    }))
    assert_camel_case(first)

    assert first.get("score") is not None, f"应该给出分数：{first}"
    assert 0 <= first["score"] <= 100, f"分数应该在 0~100，实际 {first['score']}"
    # ⚠️ 评语在 analysis 里，不是顶层
    analysis = first.get("analysis") or {}
    assert analysis.get("comment"), f"应该给出评语：{first}"
    assert analysis.get("missingPoints") is not None, f"应该指出漏掉的要点：{first}"
    # 答得不错就不该拿 0 分
    assert first["score"] > 0, f"答了这么多还给 0 分，评分大概率没生效：{first}"

    # 下一题：服务端应该给出「题号最小的未作答」
    next_question = first.get("nextQuestion")
    if next_question:
        assert next_question["questionNo"] == questions[1]["questionNo"], (
            f"下一题应该是最小未作答的那道，实际给了 {next_question}"
        )


def test_blank_answer_is_accepted(api, auth, interview):
    """跳过一题是正常行为 —— 给 0 分但不报错。"""
    questions = interview["questions"]
    result = ok(api.post(f"/app/interviews/{interview['id']}/answers", headers=auth, json={
        "questionId": questions[1]["id"],
        "answer": "",
    }))
    assert result.get("score") in (None, 0), f"空回答应该是 0 分，实际 {result.get('score')}"


def test_finish_generates_report(api, auth, interview):
    """结束面试 → 报告。综合分应该和逐题分对得上。"""
    report = ok(api.post(f"/app/interviews/{interview['id']}/finish", headers=auth))
    assert_camel_case(report)

    for field in ("overallScore", "technicalScore", "communicationScore", "problemSolvingScore"):
        value = report.get(field)
        assert value is not None, f"报告缺少 {field}：{report}"
        assert 0 <= value <= 100, f"{field} 应该在 0~100，实际 {value}"

    assert report.get("reportText"), "报告要有正文"
    assert report.get("strengths") or report.get("weaknesses"), "报告要有优点或不足"

    # 逐题分（0~100）和综合分（0~100）应该在同一量纲 —— 以前踩过 0~1 和 0~100 混用
    detail = ok(api.get(f"/app/interviews/{interview['id']}", headers=auth))
    answered = [q for q in (detail.get("questions") or []) if q.get("score") is not None]
    assert answered, "报告生成后应该能看到逐题得分"
    if answered:
        average = sum(q["score"] for q in answered) / len(answered)
        assert abs(average - report["overallScore"]) <= 40, (
            f"综合分 {report['overallScore']} 和逐题平均分 {average:.1f} 差得太远 —— "
            f"要么量纲不一致，要么模型没拿到逐题分"
        )


def test_expected_points_visible_after_finish(api, auth, interview):
    """结束后才放出参考答案要点 —— 这时用户已经答完了。"""
    detail = ok(api.get(f"/app/interviews/{interview['id']}", headers=auth))
    questions = detail.get("questions") or []
    assert any(q.get("expectedPoints") for q in questions), (
        "结束后应该能看到参考答案要点，否则用户没法复盘"
    )
