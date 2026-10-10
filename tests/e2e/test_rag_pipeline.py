"""真实链路：知识库 → 上传文档 → 等解析/切分/向量化 → 检索 → 对话（同步 + SSE）。

这一段最容易出「静默失败」：文档状态显示成功、但向量库里没有内容，
检索永远返回空 —— 接口全 200，功能是坏的。所以要真的问一个只有文档里才有答案的问题。
"""

import json
import time

import pytest

from tests.e2e.helpers import SAMPLE_DOC, assert_camel_case, ok

pytestmark = pytest.mark.e2e

# 文档处理是异步的（Python 侧轮询认领任务），给足时间
PROCESS_TIMEOUT_SECONDS = 240


def wait_until_vectorized(api, auth, document_id: int) -> dict:
    """轮询文档状态，直到向量化完成。

    失败时直接 fail 并带上 errorMessage —— 否则后面所有用例都会以
    「检索返回空」的形式失败，完全看不出真因。
    """
    deadline = time.time() + PROCESS_TIMEOUT_SECONDS
    last: dict = {}
    while time.time() < deadline:
        last = ok(api.get(f"/app/documents/{document_id}/status", headers=auth))
        vector_status = last.get("vectorStatus")
        if vector_status == "SUCCESS":
            return last
        if "FAILED" in {last.get("parseStatus"), last.get("chunkStatus"), vector_status}:
            pytest.fail(f"文档处理失败：{last}")
        time.sleep(3)
    pytest.fail(f"等文档处理超时（{PROCESS_TIMEOUT_SECONDS}s），最后状态：{last}")


@pytest.fixture(scope="module")
def knowledge_base(api, auth, cleanup):
    """建一个知识库，上传一份说明文档，等它处理完。"""
    kb = ok(api.post("/app/knowledge-bases", headers=auth, json={
        "name": "e2e 知识库",
        "description": "端到端测试用，跑完会删掉",
    }))
    kb_id = kb["id"]
    # ⚠️ 走 API 删 —— SQL 直删会绕过「删知识库 → 触发向量清理」，向量库里留孤儿分块
    cleanup.append((f"/app/knowledge-bases/{kb_id}", f"知识库 {kb_id}"))

    document = ok(api.post(
        f"/app/knowledge-bases/{kb_id}/documents",
        headers=auth,
        files={"file": ("e2e-平台说明.txt", SAMPLE_DOC.encode("utf-8"), "text/plain")},
    ))
    # ⚠️ DocumentVO 的主键字段叫 documentId，不是 id
    document_id = document["documentId"]

    status = wait_until_vectorized(api, auth, document_id)
    return {"id": kb_id, "documentId": document_id, "status": status}


def test_document_pipeline_completes(knowledge_base):
    status = knowledge_base["status"]
    assert status["parseStatus"] == "SUCCESS"
    assert status["chunkStatus"] == "SUCCESS"
    assert status["vectorStatus"] == "SUCCESS"


def test_knowledge_base_stats_reflects_document(api, auth, knowledge_base):
    stats = ok(api.get(f"/app/knowledge-bases/{knowledge_base['id']}/stats", headers=auth))
    assert_camel_case(stats)
    # 统计口径可能是 documentCount / totalDocuments，两者都认
    counts = [v for k, v in stats.items() if "document" in k.lower() and isinstance(v, int)]
    assert counts, f"stats 里应该有文档数：{stats}"
    assert max(counts) >= 1, f"刚上传了 1 篇文档，统计却是 0：{stats}"


def test_search_finds_relevant_chunk(api, auth, knowledge_base):
    """★ 检索必须真的召回内容 —— 这里失败通常意味着向量化是假成功。"""
    sources = ok(api.post("/app/rag/search", headers=auth, json={
        "knowledgeBaseId": knowledge_base["id"],
        "query": "上传文档后需要多久才能开始对话？",
        "topK": 5,
    }))
    assert sources, "检索应该召回至少一段内容"
    assert_camel_case(sources)

    # ⚠️ SourceVO 刻意不带 content —— 引用溯源要按 chunkId 单独取（见 ChunkController 的说明）。
    #    这样列表接口不用把大段原文一起吐出来。
    top = sources[0]
    assert top.get("chunkId"), f"引用要带 chunkId（前端靠它取原文）：{top}"
    assert top.get("documentName"), f"引用要带文档名（前端要显示「来自哪篇」）：{top}"


def test_chunk_can_be_fetched_for_citation(api, auth, knowledge_base):
    """引用溯源：按 chunkId 能取到原文。"""
    sources = ok(api.post("/app/rag/search", headers=auth, json={
        "knowledgeBaseId": knowledge_base["id"],
        "query": "支持哪些文档格式？",
        "topK": 3,
    }))
    assert sources, "检索应该召回内容"

    chunk = ok(api.get(f"/app/chunks/{sources[0]['chunkId']}", headers=auth))
    assert chunk.get("content"), f"按 chunkId 应该能取到原文：{chunk}"


def test_chat_answers_from_document(api, auth, cleanup, knowledge_base):
    """★ 同步对话：答案要来自文档，并带引用。"""
    conversation = ok(api.post("/app/conversations", headers=auth, json={
        "knowledgeBaseId": knowledge_base["id"],
        "title": "e2e 对话",
    }))
    conversation_id = conversation["conversationId"]
    cleanup.append((f"/app/conversations/{conversation_id}", f"会话 {conversation_id}"))

    result = ok(api.post(f"/app/conversations/{conversation_id}/messages", headers=auth, json={
        "message": "上传文档后大概要多久才能开始对话？",
        "topK": 5,
    }))
    assert_camel_case(result)

    answer = result.get("answer") or ""
    assert answer, "应该有回答"
    assert "没有找到" not in answer, (
        f"文档里明确写了这个问题的答案，却答「没找到」—— 多半是检索链路断了。回答：{answer[:200]}"
    )
    assert result.get("sources"), f"基于文档的回答应该带引用来源：{result}"


def test_chat_stream_emits_expected_events(api, auth, cleanup, knowledge_base):
    """★ SSE：事件序列必须是 sources → delta × N → done。

    用 httpx 的流式接口而不是 EventSource —— 提问要带 body，
    原生 EventSource 只支持 GET。
    """
    conversation = ok(api.post("/app/conversations", headers=auth, json={
        "knowledgeBaseId": knowledge_base["id"],
        "title": "e2e 流式对话",
    }))
    conversation_id = conversation["conversationId"]
    cleanup.append((f"/app/conversations/{conversation_id}", f"会话 {conversation_id}"))

    events: list[tuple[str, dict]] = []
    with api.stream(
        "POST",
        f"/app/conversations/{conversation_id}/messages/stream",
        headers=auth,
        json={"message": "支持哪些文档格式？", "topK": 5},
        timeout=180.0,
    ) as response:
        assert response.status_code == 200, (
            f"开流之后状态码就锁死 200 了；这里非 200 说明是开流【之前】的错误："
            f"HTTP {response.status_code} {response.read()[:200]}"
        )
        current_event = None
        for line in response.iter_lines():
            if line.startswith("event:"):
                current_event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[len("data:"):].strip())
                events.append((current_event or "message", payload))

    names = [name for name, _ in events]
    assert "sources" in names, f"流里应该有 sources 事件（先推引用，用户不用干等）：{names}"
    assert names.count("delta") > 0, f"流里应该有 delta 事件（正文分片）：{names}"
    assert names[-1] in ("done", "error"), f"流应该以 done 或 error 收尾，实际 {names[-1]}"
    assert "error" not in names, f"这次不该出错：{[p for n, p in events if n == 'error']}"

    # sources 必须在第一个 delta 之前 —— 这是「先出引用」这个设计的意义所在
    assert names.index("sources") < names.index("delta"), (
        f"sources 要先于 delta 到达，实际顺序 {names}"
    )

    done = next(payload for name, payload in events if name == "done")
    assert done.get("answer"), f"done 事件要带完整回答：{done}"
    assert done.get("messageId"), f"done 事件要带落库后的消息 id：{done}"
