"""内部接口 —— 只给 Java 后端调用。

对齐文档 §十七「阶段 9：AI 内部服务接口」。

⚠️ 权限边界（文档 §十七 的硬性要求）
本服务不负责登录、不验 JWT、不做 RBAC、不自行判断用户有没有权限。
Java 在调过来之前已经完成了权限检查，并把确认过的 userId / 知识库 id 传进来。
本服务只负责「在给定的知识库范围内算」。

响应格式对齐 Java 的 Result：{code, message, data}
这样 Java 侧不用为「调 Python」单独写一套解析。
"""

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.security import verify_service_token
from app.services import chat, interview, job, match, rag, resume
from app.services.downloader import download_bytes
from app.services.embedding import EmbeddingCallError
from app.services.llm import LlmCallError
from app.services.parser import parse

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
    enableRerank: bool = Field(default=False, description="是否用 rerank 模型精排。未配置 RERANK 模型或调用失败时降级回向量序，响应里的 reranked 字段如实反映是否生效")


@router.post("/rag/search", summary="知识库语义检索")
async def rag_search(req: RagSearchRequest) -> dict:
    """在指定知识库内做语义检索。

    Java 侧的调用时机：用户提问 → Java 查权限 → 调这个接口拿相关片段 →
    自己组装 prompt 或把片段交给 /rag/chat。
    """
    results, reranked = await rag.search(
        knowledge_base_id=req.knowledgeBaseId,
        query=req.query,
        top_k=req.topK,
        enable_rerank=req.enableRerank,
    )

    logger.info("检索请求：user=%s kb=%s topK=%d → %d 条 reranked=%s",
                req.userId, req.knowledgeBaseId, req.topK, len(results), reranked)

    return {
        "code": 0,
        "message": "成功",
        "data": {
            "results": results,
            "total": len(results),
            # 「请求了重排」不等于「重排生效了」：没配 RERANK 模型或调用失败时
            # 会降级回向量序，这里如实返回，Java 拿它决定要不要展示「已精排」标记。
            "reranked": reranked,
        },
    }


class ChatHistoryItem(BaseModel):
    """一轮历史消息。"""

    role: str = Field(description="user 或 assistant")
    content: str


class RagChatRequest(BaseModel):
    """对话请求。字段名同样是 camelCase，对齐文档 §十七。"""

    userId: str = Field(description="调用方已确认的用户 id（本服务只用于日志，不校验权限）")
    knowledgeBaseId: str | None = Field(default=None, description="知识库 id（可选，为空或 0 表示通用 AI 问答）")
    conversationId: str | None = Field(default=None, description="会话 id（仅用于日志）")
    query: str = Field(min_length=1, description="用户问题")
    topK: int = Field(default=8, ge=1, le=50, description="检索片段数")
    enableRerank: bool = Field(default=False, description="是否用 rerank 模型精排。未配置 RERANK 模型或调用失败时降级回向量序，响应里的 reranked 字段如实反映是否生效")
    promptId: str | None = Field(
        default=None,
        description="回答风格 id（用户在界面上选的，见 GET /app/prompts）。"
                    "不传 = 默认风格；传了但找不到 / 没开放给用户切换时也回退默认 —— "
                    "用户上次选的风格被管理员下线了，不该让他的提问直接失败。",
    )
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
        prompt_id=req.promptId,
        enable_rerank=req.enableRerank,
    )

    logger.info("对话请求：user=%s kb=%s conv=%s → hasAnswer=%s，%d 个引用",
                req.userId, req.knowledgeBaseId, req.conversationId,
                result["hasAnswer"], len(result["sources"]))

    return {"code": 0, "message": "成功", "data": result}


# ---------- SSE 流式对话 ----------
#
# ★ 和上面 /rag/chat 的关系：走的是同一个逻辑（chat.chat_stream），只是输出方式不同。
#   前端要「打字机效果」就走这个；要一次性拿完整结果就走上面那个。
#
# ⚠️ 两个必须知道的约束：
#   ① HTTP 状态码【只能是 200】—— 响应头在第一个字节发出时就定死了。
#      所以「检索失败」「模型报错」这些都没法用状态码表达，统一走 error 事件。
#      **前端必须处理 error 事件**，否则出错了它会一直转圈。
#   ② 只有【对话】做流式。解析 / 分析类（resumes/parse、interviews/*）不做 ——
#      它们的输出是一整个 JSON，流到一半的 JSON 没法用。


def _sse(event: str, payload: dict) -> str:
    """拼一条 SSE 消息。

    ⚠️ data 里【也带了 type 字段】，和 event 名重复。这是刻意的：
    Java 侧是手动逐行读这个流的（RestClient 不认 SSE），
    看 data 里的 type 比解析 event 字段省事；而且有些客户端只认 data。
    """
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/rag/chat/stream", summary="基于知识库的问答（SSE 流式）")
async def rag_chat_stream(req: RagChatRequest) -> StreamingResponse:
    """流式版 /rag/chat。返回 `text/event-stream`，逐段推送。

    事件序列：

    ```
    event: sources   data: {"type","sources","hasContext","contextTruncated"}
    event: delta     data: {"type","text"}              ← 很多个
    event: done      data: {"type","answer","hasAnswer"}
    event: error     data: {"type","code","message"}    ← 出问题时【代替】done
    ```

    ⚠️ 出问题时一定是「先 200，再发 error 事件」—— 见上面第 ① 条约束。
    """

    async def event_source():
        try:
            async for evt in chat.chat_stream(
                knowledge_base_id=req.knowledgeBaseId,
                query=req.query,
                top_k=req.topK,
                history=[h.model_dump() for h in req.history],
                prompt_id=req.promptId,
                enable_rerank=req.enableRerank,
            ):
                yield _sse(evt["type"], evt)

        except ValueError as exc:
            # 参数问题（查询为空之类）—— pydantic 理论上已经挡住了，兜个底
            logger.warning("流式对话参数不合法：%s", exc)
            yield _sse("error", {"type": "error", "code": 40001, "message": str(exc)})

        except EmbeddingCallError as exc:
            # ★ 这个分支是【检索阶段】失败的 —— 嵌入查询词是检索的第一步，
            #   它挂了就连引用片段都拿不到，所以此时还没发过 sources 事件。
            logger.error("流式对话向量化失败：%s", exc)
            yield _sse("error", {
                "type": "error",
                "code": 50202,
                "message": f"检索失败（向量化不可用）：{exc}",
            })

        except LlmCallError as exc:
            # ⚠️ 模型可能是【吐到一半才失败】的，这时前端已经渲染了部分文本。
            #    message 要说清「已输出的那部分可能不完整」，
            #    否则用户会把半截回答当成完整答案。
            logger.error("流式对话模型调用失败：%s", exc)
            yield _sse("error", {
                "type": "error",
                "code": 50202,
                "message": f"AI 回答中断：{exc}（已输出的部分可能不完整）",
            })

        except Exception as exc:                     # noqa: BLE001
            logger.exception("流式对话出现未预期错误")
            yield _sse("error", {
                "type": "error",
                "code": 50000,
                "message": f"服务内部错误：{type(exc).__name__}",
            })

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            # ⚠️ 这三个头都不能少：
            #   no-cache           别让浏览器 / 代理缓存这条流
            #   X-Accel-Buffering  让 nginx 不要缓冲 —— 不设的话它会攒够一批才发，
            #                      流式就白做了（本地直连看不出，一上 nginx 就露馅）
            #   keep-alive         明确告诉中间层这是长连接
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------- 简历 ----------


class ResumeParseRequest(BaseModel):
    """简历解析请求。

    Java 侧已经把权限校验做完了（简历得是他的），这里只负责算。
    """

    userId: str = Field(description="调用方已确认的用户 id（本服务只用于日志，不校验权限）")
    resumeId: str = Field(description="简历 id（仅用于日志）")
    fileUrl: str = Field(min_length=1, description="简历文件的预签名下载地址（Java 签发）")
    fileType: str = Field(min_length=1, description="文件类型：PDF / DOCX / TXT / MD / HTML")
    targetPosition: str | None = Field(
        default=None, description="用户填的目标职位，可空 —— 有的话能帮模型填对 targetPosition")


@router.post("/resumes/parse", summary="解析简历为结构化数据")
async def resume_parse(req: ResumeParseRequest) -> dict:
    """下载简历文件 → 提取文本 → 让模型结构化成 JSON。

    ⚠️ 为什么返回 rawText：
    文本提取（PDF/DOCX 解析）只有本服务会做，Java 侧没有解析器。
    但 raw_content 要存进数据库（供后续匹配、优化复用，避免每次重新下载解析），
    所以提取出来的原文必须回传给 Java。

    ⚠️ 这是个【同步】接口，一次调用可能要几十秒（下载 + 提取 + 模型生成）。
    Java 侧的读超时是 120 秒，够用。
    """
    content = await download_bytes(req.fileUrl)
    extracted = parse(content, req.fileType)

    parsed = await resume.parse_resume(
        extracted.text,
        target_position=req.targetPosition,
    )

    logger.info("简历解析：user=%s resume=%s → %d 字符、%d 项技能",
                req.userId, req.resumeId, len(extracted.text), len(parsed["skills"]))

    return {
        "code": 0,
        "message": "成功",
        "data": {
            "rawText": extracted.text,
            "pageCount": extracted.page_count,
            "summary": parsed["summary"],
            "structured": parsed["structured"],
            "skills": parsed["skills"],
            "truncated": parsed["truncated"],
        },
    }


# ---------- 职位 JD ----------


class JobParseRequest(BaseModel):
    """JD 解析请求。

    ⚠️ 和简历不同，这里传的是【文本】而不是文件地址：
    JD 本来就是用户粘贴的一段文字（存在 job.raw_content 里），
    Java 直接带过来就行，不用下载、不用文本提取。
    """

    userId: str = Field(description="调用方已确认的用户 id（本服务只用于日志，不校验权限）")
    jobId: str = Field(description="职位 id（仅用于日志）")
    text: str = Field(min_length=1, description="JD 原文")


@router.post("/jobs/parse", summary="解析职位 JD 为结构化数据")
async def job_parse(req: JobParseRequest) -> dict:
    """把 JD 文本结构化成技能 / 经验 / 学历 / 职责。

    文档 §十七 的内部接口清单里没列这个（只列了 Resume AI 和 Interview AI），
    但阶段 12 的 POST /api/app/jobs/{id}/parse 需要它，所以补上。
    """
    parsed = await job.parse_job(req.text)

    logger.info("JD 解析：user=%s job=%s → %d 项技能、%d 条职责",
                req.userId, req.jobId,
                len(parsed["skills"]), len(parsed["structured"]["responsibilities"]))

    return {
        "code": 0,
        "message": "成功",
        "data": {
            "summary": parsed["summary"],
            "structured": parsed["structured"],
            "skills": parsed["skills"],
            "truncated": parsed["truncated"],
        },
    }


# ---------- 简历分析（优化建议）----------


class ResumeAnalyzeRequest(BaseModel):
    """简历分析请求。

    ⚠️ 传【文本】而不是 resumeId：本服务不连数据库（数据访问全在 Java），
    Java 把已经存好的 resume.raw_content / job.raw_content 带过来就行。
    """

    userId: str = Field(description="调用方已确认的用户 id（本服务只用于日志，不校验权限）")
    resumeId: str = Field(description="简历 id（仅用于日志）")
    resumeText: str = Field(min_length=1, description="简历原文")
    jobText: str | None = Field(
        default=None, description="目标岗位 JD 原文，可空（不针对具体岗位时按通用标准评）")
    target: str | None = Field(
        default=None,
        description="本次重点分析的部分：SUMMARY / WORK_EXPERIENCE / "
                    "PROJECT_EXPERIENCE / SKILLS / EDUCATION / OVERALL，可空 = 整体",
    )


@router.post("/resumes/analyze", summary="针对岗位给出简历改进建议")
async def resume_analyze(req: ResumeAnalyzeRequest) -> dict:
    """给改进建议 —— <b>只给建议，不改写简历本身，也不落库</b>。

    ⚠️ 返回值里 `needs_fact=true` 的条目表示「这里缺事实，需要用户自己补」——
    前端**要把它和普通改写建议分开展示**，不能混在一起。
    原因见 prompts/resume_analyze.txt 开头那段：编造出来的经历
    会让用户在面试时被追问两句就露馅，比不改还糟。
    """
    result = await resume.analyze_resume(
        req.resumeText,
        job_text=req.jobText,
        target=req.target,
    )

    need_fact = sum(1 for i in result["items"] if i["needsFact"])
    logger.info("简历分析：user=%s resume=%s → %d 条建议（%d 条待补事实）",
                req.userId, req.resumeId, len(result["items"]), need_fact)

    return {"code": 0, "message": "成功", "data": result}


# ---------- 简历-JD 匹配 ----------


class ResumeMatchRequest(BaseModel):
    """简历-JD 匹配请求。

    ⚠️ 和 resumes/analyze 一样传【文本】而不是 id：本服务不连数据库（数据访问全在 Java），
    Java 把已经存好的 resume.raw_content / job.raw_content 带过来就行。
    """

    userId: str = Field(description="调用方已确认的用户 id（本服务只用于日志，不校验权限）")
    resumeId: str = Field(description="简历 id（仅用于日志）")
    jobId: str = Field(description="职位 id（仅用于日志）")
    resumeText: str = Field(min_length=1, description="简历原文")
    jobText: str = Field(min_length=1, description="目标岗位 JD 原文")


@router.post("/resumes/match", summary="简历与职位匹配分析（五类结论）")
async def resume_match(req: ResumeMatchRequest) -> dict:
    """逐条对照 JD 给出匹配结论。

    ★ 每条要求落到五类之一：已匹配 / 表达缺口 / 证据不足 / 真实缺口 / 待确认。
    `overallScore` 只作展示辅助，**不参与任何准入判断** ——
    百分比没法行动，五类才能直接告诉用户该改哪一句、补哪件事。

    ⚠️ 和 /resumes/analyze 的区别：那个是「这份简历自己怎么写得更好」（不需要参照物），
    这个是「对【这个岗位】够不够」（必须有 JD 才成立）。两者不能互相替代。
    """
    result = await match.match_resume(req.resumeText, req.jobText)

    counts: dict[str, int] = {}
    for item in result["items"]:
        counts[item["verdict"]] = counts.get(item["verdict"], 0) + 1

    logger.info("简历匹配：user=%s resume=%s job=%s → %d 条要求 %s",
                req.userId, req.resumeId, req.jobId, len(result["items"]), counts)

    return {"code": 0, "message": "成功", "data": result}


# ---------- AI 面试 ----------


class InterviewGenerateRequest(BaseModel):
    """出题请求。"""

    userId: str = Field(description="调用方已确认的用户 id（本服务只用于日志，不校验权限）")
    interviewId: str = Field(description="面试 id（仅用于日志）")
    interviewType: str = Field(
        default="TECHNICAL", description="TECHNICAL / BEHAVIORAL / MIXED")
    difficulty: str = Field(default="MEDIUM", description="EASY / MEDIUM / HARD")
    questionCount: int = Field(default=5, ge=1, le=20, description="题目数量")
    resumeText: str | None = Field(default=None, description="候选人简历原文，可空")
    jobText: str | None = Field(default=None, description="目标岗位 JD 原文，可空")


@router.post("/interviews/generate", summary="根据简历和 JD 出面试题")
async def interview_generate(req: InterviewGenerateRequest) -> dict:
    """一次性出完所有题目。

    ⚠️ 每题都带 `expectedPoints`（参考答案要点）—— Java 要把它存进
    `interview_question.expected_points`，后面调 /interviews/analyze 时再带回来。
    **没有这个参照物，模型对同一份回答的打分会飘**（这次 70 下次 85）。
    """
    result = await interview.generate_questions(
        resume_text=req.resumeText,
        job_text=req.jobText,
        interview_type=req.interviewType,
        difficulty=req.difficulty,
        question_count=req.questionCount,
    )

    logger.info("面试出题：user=%s interview=%s → %d 道题",
                req.userId, req.interviewId, len(result["questions"]))

    return {"code": 0, "message": "成功", "data": result}


class InterviewAnalyzeRequest(BaseModel):
    """单题评分请求。"""

    userId: str = Field(description="调用方已确认的用户 id（本服务只用于日志，不校验权限）")
    interviewId: str = Field(description="面试 id（仅用于日志）")
    question: str = Field(min_length=1, description="题目")
    answer: str = Field(
        default="",
        description="候选人的回答。⚠️ 允许空 —— 跳过一道题是正常行为，应该给低分而不是报错",
    )
    expectedPoints: list[str] = Field(
        default_factory=list, description="参考答案要点（出题时存下来的），建议带上")
    interviewType: str = Field(default="TECHNICAL", description="面试类型")


@router.post("/interviews/analyze", summary="评估候选人对单道题的回答")
async def interview_analyze(req: InterviewAnalyzeRequest) -> dict:
    """给一道题的回答打分 + 分析。

    ⚠️ 空回答【不报错】—— 返回一个低分结果，让流程继续走完。
    报错的话用户跳过一道题就整场面试卡住了。
    """
    result = await interview.analyze_answer(
        question=req.question,
        answer=req.answer,
        expected_points=req.expectedPoints,
        interview_type=req.interviewType,
    )

    logger.info("答题分析：user=%s interview=%s → 得分 %s，漏 %d 个要点",
                req.userId, req.interviewId, result["score"],
                len(result["missingPoints"]))

    return {"code": 0, "message": "成功", "data": result}


class InterviewReportItem(BaseModel):
    """报告里的单题记录。"""

    questionNo: int = Field(description="题号")
    question: str = Field(description="题目")
    answer: str | None = Field(default=None, description="候选人的回答")
    score: float | None = Field(default=None, description="本题得分")
    analysis: dict | None = Field(
        default=None,
        description="本题的分析结果（/interviews/analyze 的返回）。"
                    "⚠️ 建议带上 —— 报告要求「四个分数和逐题得分对得上」，"
                    "没有逐题分模型只能瞎给，用户会发现「每题都 80 综合却给 60」",
    )


class InterviewReportRequest(BaseModel):
    """总报告请求。"""

    userId: str = Field(description="调用方已确认的用户 id（本服务只用于日志，不校验权限）")
    interviewId: str = Field(description="面试 id（仅用于日志）")
    interviewType: str = Field(default="TECHNICAL", description="面试类型")
    jobTitle: str | None = Field(default=None, description="目标岗位名，可空")
    items: list[InterviewReportItem] = Field(
        min_length=1, description="逐题记录，至少一条")


@router.post("/interviews/report", summary="根据整场问答生成复盘报告")
async def interview_report(req: InterviewReportRequest) -> dict:
    """生成复盘报告。

    返回的字段直接对齐 `interview_report` 表：
    四个分数 + strengths/weaknesses/suggestions 三个数组 + reportText（Markdown）。
    """
    result = await interview.build_report(
        items=[item.model_dump() for item in req.items],
        interview_type=req.interviewType,
        job_title=req.jobTitle,
    )

    logger.info("面试报告：user=%s interview=%s → 综合 %s（%d 道题）",
                req.userId, req.interviewId, result["overallScore"], len(req.items))

    return {"code": 0, "message": "成功", "data": result}


# ---------- 网页正文提取工具 ----------


class UrlExtractRequest(BaseModel):
    """网页提取请求。"""

    url: str = Field(min_length=1, description="目标网页 URL")


@router.post("/tools/url-extract", summary="网页正文高纯度提取为 Markdown")
async def tool_url_extract(req: UrlExtractRequest) -> dict:
    """提取网页正文并转换为 Markdown。"""
    from app.services.tools.web_extractor import extract_url

    try:
        data = await extract_url(req.url)
        return {"code": 0, "message": "成功", "data": data}
    except ValueError as exc:
        return {"code": 40001, "message": str(exc), "data": None}
    except Exception as exc:
        logger.exception("网页提取失败: %s", exc)
        return {"code": 50000, "message": f"网页提取失败: {exc}", "data": None}
