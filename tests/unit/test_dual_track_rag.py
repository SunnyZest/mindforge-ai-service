"""公域官方智库 + 私域个人库双路联合检索（Dual-Track RAG）与模拟面试/简历赋能单元测试。"""

import json
import pytest

from app.services import interview, resume
from app.services.vector_store import VectorStore


class TestRequireKbIds:
    """校验 VectorStore 多知识库 ID 解析与防泄漏安全边界。"""

    def test_single_id(self):
        assert VectorStore._require_kb_ids("3") == [3]
        assert VectorStore._require_kb_ids(6) == [6]

    def test_comma_separated_ids_deduplicated_preserving_order(self):
        assert VectorStore._require_kb_ids("3, 6, 3, 8") == [3, 6, 8]

    def test_list_of_ids(self):
        assert VectorStore._require_kb_ids([3, "6", 3]) == [3, 6]

    @pytest.mark.parametrize("bad", [None, "", "   ", " , ", "abc", "3,abc"])
    def test_invalid_or_empty_raises(self, bad):
        with pytest.raises(ValueError):
            VectorStore._require_kb_ids(bad)


@pytest.mark.anyio
async def test_interview_generate_injects_official_kb_references(monkeypatch):
    """出题时传入官方智库 ID，应自动召回参考资料并注入面试官 Prompt。"""
    captured_prompts: list[str] = []

    async def fake_search(*, knowledge_base_id, query, top_k=None, min_score=None, enable_rerank=False):
        assert knowledge_base_id == "6"
        assert "Java" in query or "Redis" in query
        return (
            [
                {
                    "chunkId": "doc10-chunk0",
                    "documentId": 10,
                    "knowledgeBaseId": 6,
                    "chunkIndex": 0,
                    "content": "大厂高频真题：Redis 分布式锁 Redlock 争议点与看门狗续期机制。",
                    "score": 0.88,
                }
            ],
            False,
        )

    async def fake_llm_chat(messages, **kwargs):
        captured_prompts.append(messages[0]["content"])
        return json.dumps({
            "questions": [
                {
                    "questionNo": 1,
                    "questionType": "TECHNICAL",
                    "question": "结合你在订单系统使用的 Redis 分布式锁，谈谈看门狗续期与主从切换锁丢失怎么应对？",
                    "expectedPoints": ["Watchdog 自动续期", "主从异步复制导致锁丢失", "Redlock 或业务幂等兜底"],
                    "difficulty": "HARD",
                }
            ]
        })

    monkeypatch.setattr(interview.rag, "search", fake_search)
    monkeypatch.setattr(interview.llm_service, "chat", fake_llm_chat)

    res = await interview.generate_questions(
        resume_text="负责高并发订单系统，使用 Java 与 Redis 实现分布式锁防超卖。",
        job_text="招聘高级 Java 工程师，熟悉 Redis 高可用架构。",
        interview_type="TECHNICAL",
        difficulty="HARD",
        question_count=1,
        knowledge_base_id="6",
    )

    assert len(res["questions"]) == 1
    assert captured_prompts
    assert "【官方智库高频真题与考核标准参考】" in captured_prompts[0]
    assert "Redlock 争议点与看门狗续期机制" in captured_prompts[0]


@pytest.mark.anyio
async def test_interview_generate_degrades_gracefully_on_rag_error(monkeypatch):
    """官方智库检索异常时静默降级，不阻断面试出题主流程。"""
    async def broken_search(**kwargs):
        raise RuntimeError("向量库连接繁忙")

    async def fake_llm_chat(messages, **kwargs):
        return json.dumps({
            "questions": [
                {
                    "questionNo": 1,
                    "questionType": "TECHNICAL",
                    "question": "说说 JVM 垃圾回收器？",
                    "expectedPoints": ["G1", "ZGC"],
                    "difficulty": "MEDIUM",
                }
            ]
        })

    monkeypatch.setattr(interview.rag, "search", broken_search)
    monkeypatch.setattr(interview.llm_service, "chat", fake_llm_chat)

    res = await interview.generate_questions(
        resume_text="Java 后端开发",
        job_text="Java 工程师",
        question_count=1,
        knowledge_base_id="6",
    )
    assert len(res["questions"]) == 1


@pytest.mark.anyio
async def test_resume_analyze_injects_official_kb_references(monkeypatch):
    """简历润色传入官方智库 ID 时，应自动召回行业标杆参考并注入 Prompt。"""
    from app.services import rag as rag_module

    captured_prompts: list[str] = []

    async def fake_search(*, knowledge_base_id, query, top_k=None, min_score=None, enable_rerank=False):
        assert knowledge_base_id == "6"
        return (
            [
                {
                    "chunkId": "doc11-chunk0",
                    "documentId": 11,
                    "knowledgeBaseId": 6,
                    "chunkIndex": 0,
                    "content": "标杆简历要求：微服务改造需量化 QPS 提升比例与 P99 延迟下降数据。",
                    "score": 0.85,
                }
            ],
            False,
        )

    async def fake_llm_chat(messages, **kwargs):
        captured_prompts.append(messages[0]["content"])
        return json.dumps({
            "overallScore": 78,
            "summary": "项目经历缺乏量化指标",
            "highlights": ["技术栈匹配"],
            "items": [
                {
                    "target": "PROJECT_EXPERIENCE",
                    "original": "负责订单服务重构",
                    "issue": "缺乏性能提升的量化结果",
                    "suggestion": "补充 QPS 及 P99 延迟变化",
                    "rewritten": None,
                    "needsFact": True,
                    "missingFact": "请补充重构前后的 QPS 与接口耗时对比数据",
                }
            ],
        })

    monkeypatch.setattr(rag_module, "search", fake_search)
    monkeypatch.setattr(resume.llm_service, "chat", fake_llm_chat)

    res = await resume.analyze_resume(
        "个人简历：负责订单服务重构，使用 Spring Cloud。",
        job_text="要求熟悉高并发微服务调优",
        target="PROJECT_EXPERIENCE",
        knowledge_base_id="6",
    )

    assert len(res["items"]) == 1
    assert captured_prompts
    assert "【官方智库行业标杆与技术要点参考】" in captured_prompts[0]
    assert "量化 QPS 提升比例与 P99 延迟下降数据" in captured_prompts[0]
