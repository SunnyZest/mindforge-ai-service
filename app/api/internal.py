"""内部接口 —— 只给 Java 后端调用。

对齐文档 §十七「阶段 9：AI 内部服务接口」。

⚠️ 权限边界（文档 §十七 的硬性要求）
本服务不负责登录、不验 JWT、不做 RBAC、不自行判断用户有没有权限。
Java 在调过来之前已经完成了权限检查，并把确认过的 userId / 知识库 id 传进来。
本服务只负责「在给定的知识库范围内算」。

响应格式对齐 Java 的 Result：{code, message, data}
这样 Java 侧不用为「调 Python」单独写一套解析。
"""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import verify_service_token
from app.services import chat, rag

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/internal/ai/v1",
    tags=["内部接口（仅 Java 调用）"],
    # 整个 router 统一校验 X-Service-Token，不用每个接口各写一遍
    dependencies=[Depends(verify_service_token)],
)


class RagSearchRequest(BaseModel):
    """检索请求。

    字段名用 camelCase 是刻意的 —— 直接对齐文档 §十七 和 Java 的命名，
    省掉两边各转一次。这是接口契约，不是 Python 内部数据结构。
    """

    userId: str = Field(description="调用方已确认的用户 id（本服务只用于日志，不校验权限）")
    knowledgeBaseId: str = Field(description="知识库 id —— 数据隔离的安全边界")
    query: str = Field(min_length=1, description="用户问题")
    topK: int = Field(default=8, ge=1, le=50, description="返回片段数")
    enableRerank: bool = Field(default=False, description="是否重排序")


@router.post("/rag/search", summary="知识库语义检索")
async def rag_search(req: RagSearchRequest) -> dict:
    """在指定知识库内做语义检索。

    Java 侧的调用时机：用户提问 → Java 查权限 → 调这个接口拿相关片段 →
    自己组装 prompt 或把片段交给 /rag/chat。
    """
    results = await rag.search(
        knowledge_base_id=req.knowledgeBaseId,
        query=req.query,
        top_k=req.topK,
    )

    logger.info("检索请求：user=%s kb=%s topK=%d → %d 条",
                req.userId, req.knowledgeBaseId, req.topK, len(results))

    return {
        "code": 0,
        "message": "成功",
        "data": {
            "results": results,
            "total": len(results),
            # enableRerank 目前不生效（没有接重排序模型）。
            # 这里如实返回 false，而不是假装重排过了。
            "reranked": False,
        },
    }


class ChatHistoryItem(BaseModel):
    """一轮历史消息。"""

    role: str = Field(description="user 或 assistant")
    content: str


class RagChatRequest(BaseModel):
    """对话请求。字段名同样是 camelCase，对齐文档 §十七。"""

    userId: str = Field(description="调用方已确认的用户 id（本服务只用于日志，不校验权限）")
    knowledgeBaseId: str = Field(description="知识库 id —— 数据隔离的安全边界")
    conversationId: str | None = Field(default=None, description="会话 id（仅用于日志）")
    query: str = Field(min_length=1, description="用户问题")
    topK: int = Field(default=8, ge=1, le=50, description="检索片段数")
    enableRerank: bool = Field(default=False, description="是否重排序")
    history: list[ChatHistoryItem] = Field(
        default_factory=list,
        description="历史消息（可选）。Java 侧从 chat_message 表取最近几轮传过来；"
                    "不传就是单轮问答。",
    )


@router.post("/rag/chat", summary="基于知识库的问答")
async def rag_chat(req: RagChatRequest) -> dict:
    """检索知识库并生成回答。

    和 /rag/search 的区别：search 只返回片段，chat 会基于片段生成一段回答。

    ⚠️ hasContext 和 hasAnswer 是两回事，前端别混用：
    - hasContext  —— 有没有检索到片段。展示「参考了 N 个片段」用它
    - hasAnswer   —— 模型有没有给出实质回答。判断「要不要显示答案」用它
    两者都是 True 是常态；hasContext=True + hasAnswer=False 也常见 ——
    检索到了片段，但片段里没有用户问的那个信息，模型如实说「资料里没有」。
    """
    result = await chat.chat(
        knowledge_base_id=req.knowledgeBaseId,
        query=req.query,
        top_k=req.topK,
        history=[h.model_dump() for h in req.history],
    )

    logger.info("对话请求：user=%s kb=%s conv=%s → hasAnswer=%s，%d 个引用",
                req.userId, req.knowledgeBaseId, req.conversationId,
                result["hasAnswer"], len(result["sources"]))

    return {"code": 0, "message": "成功", "data": result}
