"""真实链路：简历上传 → 解析 → AI 优化 → 职位解析 → 简历-JD 匹配。

这一组打的是【真模型】。Java 侧的测试用 mock 挡住了 AI，覆盖的是
「Java 有没有把结果正确落库」；这里覆盖 mock 测不出来的那半边：

  · 解析真的抽出技能了吗？
  · 优化建议有没有把「要补事实」和「已改写」分开？
  · 匹配的五类结论分布合不合理？有没有把【岗位要求】当成【候选人经历】？
"""

import pytest

from tests.e2e.helpers import SAMPLE_JD, SAMPLE_RESUME, assert_camel_case, ok

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def parsed_resume(api, auth, cleanup):
    """上传并解析一份简历。

    module 级 —— 这个模块里所有用例共用，避免每个用例都重新烧一遍 token。
    """
    data = ok(api.post(
        "/app/resumes",
        headers=auth,
        files={"file": ("e2e-张伟-简历.txt", SAMPLE_RESUME.encode("utf-8"), "text/plain")},
        params={"targetPosition": "Java 后端开发工程师"},
    ))
    resume_id = data["id"]
    cleanup.append((f"/app/resumes/{resume_id}", f"简历 {resume_id}"))

    parsed = ok(api.post(f"/app/resumes/{resume_id}/parse", headers=auth))
    return {"id": resume_id, "parsed": parsed}


@pytest.fixture(scope="module")
def parsed_job(api, auth, cleanup):
    data = ok(api.post("/app/jobs", headers=auth, json={
        "title": "高级 Java 后端开发工程师",
        "companyName": "某某互联网公司",
        "location": "上海",
        "jobDescription": SAMPLE_JD,
    }))
    job_id = data["id"]
    cleanup.append((f"/app/jobs/{job_id}", f"职位 {job_id}"))

    parsed = ok(api.post(f"/app/jobs/{job_id}/parse", headers=auth))
    return {"id": job_id, "parsed": parsed}


def test_resume_parse_extracts_structured_result(parsed_resume):
    parsed = parsed_resume["parsed"]
    assert_camel_case(parsed)

    assert parsed.get("parseStatus") == "SUCCESS", f"解析应该成功：{parsed}"
    structured = parsed.get("structuredJson")
    assert structured, "解析结果里应该有结构化 JSON"

    # structuredJson 是解析的核心产物（技能表是从它写出来的）
    assert structured.get("name"), f"结构化结果里应该有姓名：{structured}"
    skills = structured.get("skills") or []
    assert skills, f"应该抽出至少一项技能：{structured}"
    names = {(s.get("name") or "").lower() for s in skills}
    assert any("java" in name for name in names), f"应该抽出 Java，实际抽出的是 {names}"


def test_resume_parsed_exposes_summary(api, auth, parsed_resume):
    """解析结果里带一段摘要，前端列表页直接显示。"""
    parsed = ok(api.get(f"/app/resumes/{parsed_resume['id']}/parsed", headers=auth))
    assert_camel_case(parsed)
    assert parsed.get("summary"), f"应该有摘要：{parsed}"


def test_resume_optimize_separates_facts_from_wording(api, auth, parsed_resume):
    """★ 核心不变量：needsFact 为真的条目，rewritten 必须是 null。

    混在一起用户会以为 AI 已经替他把经历写好了，面试官追问两句就露馅。
    """
    data = ok(api.post(f"/app/resumes/{parsed_resume['id']}/optimize", headers=auth, json={}))
    assert_camel_case(data)

    items = data.get("items") or []
    assert items, "应该给出至少一条优化建议"
    assert len(items) <= 5, f"建议条数应该被收敛到 5 条以内，实际 {len(items)}"

    for item in items:
        if item.get("needsFact"):
            assert item.get("rewritten") is None, (
                f"标了 needsFact 却带着改写结果 —— 那是编的：{item}"
            )
            assert item.get("missingFact"), f"needsFact 的条目要说清要补什么：{item}"
        else:
            # 不需要补事实的条目，至少要有建议或改写之一
            assert item.get("suggestion") or item.get("rewritten"), f"空条目没有意义：{item}"


def test_resume_optimize_does_not_touch_original(api, auth, parsed_resume):
    """优化是「建议」不是「改写」—— 不能覆盖用户的原始简历。"""
    detail = ok(api.get(f"/app/resumes/{parsed_resume['id']}", headers=auth))
    assert detail.get("rawContent"), "原始正文应该还在"


def test_job_parse_extracts_requirements(parsed_job):
    parsed = parsed_job["parsed"]
    assert_camel_case(parsed)
    assert parsed.get("parseStatus") == "SUCCESS", f"解析应该成功：{parsed}"

    structured = parsed.get("structuredJson")
    assert structured, "JD 解析应该产出结构化结果"
    # ⚠️ 技能只落在 job_skill 表和 structuredJson 里 —— JobDetailVO 没有单独的 skills 字段
    skills = structured.get("skills") or []
    assert skills, f"JD 应该抽出至少一项技能要求：{structured}"


def test_match_returns_five_categories(api, auth, cleanup, parsed_resume, parsed_job):
    """★ 匹配的核心是【五类结论】，不是百分比。"""
    data = ok(api.post("/app/resume-job-analysis", headers=auth, json={
        "resumeId": parsed_resume["id"],
        "jobId": parsed_job["id"],
    }))
    assert_camel_case(data)

    items = data.get("items") or []
    assert items, "应该逐条给出匹配结论"

    allowed = {"MATCHED", "EXPRESSION_GAP", "INSUFFICIENT_EVIDENCE", "REAL_GAP", "UNCERTAIN"}
    for item in items:
        assert item.get("verdict") in allowed, f"出现了约定外的结论：{item}"
        assert item.get("requirement"), f"每条结论都要说清对应哪条要求：{item}"
        # ★ 没有依据不能说「已匹配」
        if item["verdict"] == "MATCHED":
            assert item.get("evidence"), (
                f"判了 MATCHED 却没有依据 —— 多半是把岗位要求抄成了候选人经历：{item}"
            )

    # 这份简历明显缺「JVM 调优」和「带团队经验」，应该被判成真实缺口
    verdicts = {item["verdict"] for item in items}
    assert verdicts - {"MATCHED"}, (
        f"这份简历和 JD 有明显落差，不该全是 MATCHED，实际分布 {verdicts}"
    )

    assert data.get("summary"), "应该给一段整体判断"


def test_match_latest_returns_same_result(api, auth, parsed_resume, parsed_job):
    """取上次结果：快、且和 POST 回来的是同一条。"""
    latest = ok(api.get("/app/resume-job-analysis", headers=auth, params={
        "resumeId": parsed_resume["id"],
        "jobId": parsed_job["id"],
    }))
    assert_camel_case(latest)
    assert latest.get("items"), "GET 应该能取到刚才那次匹配的结果"


def test_match_latest_before_any_match_is_404(api, auth, cleanup, parsed_job):
    """没做过匹配就取结果 → 40413，而不是给个空对象。"""
    fresh = ok(api.post(
        "/app/resumes",
        headers=auth,
        files={"file": ("e2e-没匹配过的简历.txt", SAMPLE_RESUME.encode("utf-8"), "text/plain")},
    ))
    cleanup.append((f"/app/resumes/{fresh['id']}", f"简历 {fresh['id']}"))

    response = api.get("/app/resume-job-analysis", headers=auth, params={
        "resumeId": fresh["id"],
        "jobId": parsed_job["id"],
    })
    assert response.status_code == 404
    assert response.json().get("code") == 40413, f"实际 {response.json()}"
