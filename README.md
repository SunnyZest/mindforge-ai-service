# MindForge AI Service

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/FastAPI-0.142+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/ChromaDB-1.5+-FF6F61?style=flat-square" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/DashScope-OpenAI%20Compatible-FF6A00?style=flat-square" alt="DashScope" />
</p>

MindForge AI 的 **Python AI 计算服务**：负责文档解析、分块向量化、向量检索、RAG 问答，以及简历 / 职位 / 面试等场景的结构化抽取。

它不直接对浏览器开放，只接受 Java 业务后端的调用（`X-Service-Token` 校验）。**所有数据、权限与事务都在 Java 侧**，本服务是一个无状态的计算节点。

| 我想…… | 看这里 |
| :--- | :--- |
| 在本机跑起来 | [4. 快速开始](#4-快速开始) |
| 对接接口 / 排查报错 | [6. 接口文档](#6-接口文档) |
| 部署到服务器 | [7. 部署](#7-部署) |
| 改代码 / 加接口 | [11. 开发约定](#11-开发约定) |

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 功能特性](#2-功能特性)
- [3. 系统架构](#3-系统架构)
- [4. 快速开始](#4-快速开始)
- [5. 配置说明](#5-配置说明)
- [6. 接口文档](#6-接口文档)
- [7. 部署](#7-部署)
- [8. 测试](#8-测试)
- [9. 常见问题](#9-常见问题)
- [10. 目录结构](#10-目录结构)
- [11. 开发约定](#11-开发约定)

---

## 1. 项目简介

MindForge AI 是一个面向求职者的 AI 平台（Java + Vue 双前端 + 本 Python 服务）。本服务承担其中所有「重计算、需要模型」的部分：

- 用户上传的资料（PDF / DOCX / 图片）要变成可检索的知识库 —— 解析、分块、向量化在这里做；
- 用户提问时要基于知识库回答 —— 检索、重排、生成、流式推流在这里做；
- 简历、职位 JD、面试邀约要变成结构化数据 —— 提示词与输出清洗在这里做。

**技术栈**

| 类别 | 选型 |
| :--- | :--- |
| 语言 / 运行时 | Python 3.12 |
| Web 框架 | FastAPI + Uvicorn（ASGI，全异步） |
| 模型接入 | 阿里云百炼 DashScope（OpenAI 兼容接口），`openai` SDK |
| 向量库 | ChromaDB（本地嵌入式，持久化到磁盘） |
| 文档解析 | `pdfplumber` / `pypdf` / `python-docx` / `trafilatura` / `RapidOCR` |
| 依赖管理 | `uv`（依赖声明在 `pyproject.toml`，**没有 requirements.txt**） |

---

## 2. 功能特性

| 能力 | 说明 | 实现位置 |
| :--- | :--- | :--- |
| 文档解析 | TXT / MD / PDF / DOCX / HTML / PNG / JPG 统一转成带结构标记的 Markdown 文本；扫描件与截图走 OCR | `services/parser.py` |
| 分块与向量化 | 按字符数做重叠切分，再分批调嵌入模型（单批 25 条，受接口限制） | `services/splitter.py`、`services/embedding.py` |
| 向量检索 | 所有向量存同一个 collection，用 `knowledge_base_id` 元数据过滤实现知识库隔离；可选交叉重排（Rerank），未配置或失败自动降级回向量序 | `services/vector_store.py`、`services/rag.py`、`services/rerank.py` |
| RAG 问答 | 检索 + 生成，支持多轮历史与回答风格；未召回时降级为通用问答并如实标记 | `services/chat.py` |
| 流式推流 | SSE 逐字输出，首字延迟约 0.5 秒（实测，视模型与网络而定） | `api/internal.py` |
| 结构化抽取 | 简历解析、职位 JD 解析、面试邀约抽取 —— 只抽原文有的，抽不到返回 null | `services/resume.py`、`job.py`、`invitation.py` |
| 简历分析 | 针对目标岗位给改进建议；缺事实的地方标 `needsFact`，不替用户编 | `services/resume.py` |
| 简历-JD 匹配 | 五类结论（匹配 / 部分匹配 / 不匹配 / 信息不足 / 待补充） | `services/match.py` |
| 模拟面试 | 出题、单题评分、整场复盘报告 | `services/interview.py` |
| 薪酬估算 | 给出 P50 / P75 / P90 区间，服务端做数值钳制，不美化 | `services/salary.py` |
| 网页正文提取 | `trafilatura` 提取正文，逐跳校验重定向 + 大小上限，防 SSRF | `services/tools/web_extractor.py` |
| 用量记账 | 每次模型调用上报 token 与耗时，失败只打日志、不影响业务 | `services/ai_usage.py` |

---

## 3. 系统架构

### 3.1 职责边界

```
┌──────────────────────────────┐            ┌──────────────────────────────┐
│  Java 业务后端 (Spring Boot)  │            │  Python 计算服务 (FastAPI)    │
├──────────────────────────────┤            ├──────────────────────────────┤
│ · MySQL 直连与事务            │            │ · 无状态，不连数据库          │
│ · 用户鉴权与 RBAC             │  REST/SSE  │ · 文档解析与正文抓取          │
│ · 签发 OSS 预签名 URL         │ ◄────────► │ · 分块与向量嵌入              │
│ · 会话上下文与业务状态机       │            │ · 向量索引与检索              │
└──────────────────────────────┘            │ · 模型推理与流式推流          │
                                            └──────────────────────────────┘
```

两条硬规则：

1. **不越权**：本服务不判断「这个用户能不能访问这个知识库」。权限在 Java 侧校验完才调用。
2. **不持有密钥**：下载用户文件走 Java 下发的有时效预签名 URL，本服务不需要配置对象存储的 AccessKey。

### 3.2 两条主链路

**文档入库（异步，Pull 模式）** —— 解析是长耗时任务，不做成同步接口：

```
Python 每 5 秒 ──► POST /internal/documents/claim?limit=10   Java 用 FOR UPDATE SKIP LOCKED 认领
                  ◄── 返回待处理文档（含预签名下载地址）
下载 ──► 解析 ──► 分块 ──► 嵌入 ──► 写入 Chroma
                  ──► PUT  /internal/documents/{id}/status   分阶段上报 PARSING / CHUNKING / SUCCESS
                  ──► POST /internal/documents/{id}/chunks   批量回写分块元数据
```

由 Python 主动拉取而不是 Java 推送：不受 NAT 限制，也天然形成背压（处理不过来就少认领）。

**在线问答（同步 / 流式）**：

```
浏览器 ──► Java 校验权限 ──► POST /internal/ai/v1/rag/chat/stream ──► 检索 + 生成 ──► SSE 逐段返回
```

### 3.3 配置下发

模型参数与提示词有两层来源，**管理端优先**：

- `services/ai_config.py`：每 60 秒从 Java 拉一次模型配置（chat / embedding / rerank 三个场景），拉不到就用 `.env`。
- `services/prompt_store.py`：同理拉提示词版本，拉不到就用 `prompts/*.txt`。

所以「管理端改了配置但没生效」时，先看启动日志里打印的 `AI 模型配置 : ...` 一行 —— 它写明了当前生效的是哪一套。

---

## 4. 快速开始

### 4.1 环境要求

| 依赖 | 版本 / 说明 |
| :--- | :--- |
| Python | 3.12 及以上 |
| uv | 推荐（[安装文档](https://docs.astral.sh/uv/)）；也可用原生 venv + pip |
| Java 后端 | 需要先跑起来（默认 `http://localhost:8080/api`） |
| DashScope API Key | 必须，用于对话与嵌入模型 |

### 4.2 安装依赖

```bash
# 方式 A：uv（推荐）
uv venv --python 3.12
uv sync

# 方式 B：原生 venv
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install .
```

> 仓库**没有 `requirements.txt`**，依赖统一声明在 `pyproject.toml`。新增依赖请用 `uv add xxx`，它会同时更新 `uv.lock`。

### 4.3 配置

```bash
cp .env.example .env            # Windows 用 copy
```

至少要填两项（其余可用默认值，完整清单见 [5. 配置说明](#5-配置说明)）：

```env
# 必须带 context-path /api，漏了所有回调都会 404
JAVA_BASE_URL=http://localhost:8080/api

# 必须与 Java 侧 mindforge.internal.service-token 完全一致（含大小写与末尾的 =）
JAVA_SERVICE_TOKEN=<与 Java 侧相同的令牌>

# 阿里云百炼密钥
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 4.4 启动

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 或已在虚拟环境里
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动日志会打印 Java 地址、轮询开关、分块参数、当前生效的模型与提示词来源。看到 `连不上 Java` 不影响启动，只说明 Java 还没起或令牌不对。

### 4.5 验证

| 检查项 | 方式 | 期望 |
| :--- | :--- | :--- |
| 服务存活 | `GET http://localhost:8000/health` | `{"status":"ok",...}` |
| 接口文档 | 浏览器打开 `http://localhost:8000/docs` | Swagger UI |
| 与 Java 连通 | `GET http://localhost:8000/health/java` | `{"status":"ok","javaBaseUrl":...}` |

> `/health/java` **有副作用**：它会真的调一次认领接口，可能把一篇待处理文档领走（30 分钟后 Java 会回收）。所以它只用于手动排查，监控探针请用 `/health`。

---

## 5. 配置说明

优先级：**环境变量 > `.env` > 代码默认值**。字段名大小写不敏感。

**必填**

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `JAVA_BASE_URL` | `http://localhost:8080/api` | Java 后端地址，**必须带 `/api`** |
| `JAVA_SERVICE_TOKEN` | 空 | 服务间令牌，两边必须完全一致；空值会让所有 `/internal` 请求被 Java 拒绝 |
| `DASHSCOPE_API_KEY` | 空 | 百炼密钥；管理端配了模型时可留空 |

**模型**

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI 兼容端点 |
| `CHAT_MODEL` | `deepseek-v4-flash` | 对话模型（管理端配置优先） |
| `EMBEDDING_MODEL` | `text-embedding-v1` | 嵌入模型（维度由模型决定） |
| `LLM_MAX_RETRIES` | `1` | 模型调用重试次数；**最坏耗时 = 超时 × (1 + 本值)**，Java 的读超时按这个定 |
| `EXTRACT_MAX_TOKENS` | `8000` | 结构化抽取的输出上限（简历 JSON 较大，不能用对话的 2000） |
| `EXTRACT_TIMEOUT_SECONDS` | `300` | 结构化抽取的单次超时 |
| `EXTRACT_ENABLE_THINKING` | `false` | 抽取类场景是否让模型先思考，见 [9.4](#94-简历解析或邀约抽取很慢) |
| `INVITATION_MAX_TOKENS` | `1500` | 邀约抽取的输出上限（只输出几条日程） |
| `INVITATION_TIMEOUT_SECONDS` | `55` | 邀约抽取单次超时；`55 × 2 = 110` 秒 < Java 的 115 秒读超时 |
| `INVITATION_MAX_CHARS` | `20000` | 邀约原文长度上限，超出截断 |

**对话**

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `CHAT_TEMPERATURE` | `0.3` | 问答温度，调低以减少编造 |
| `CHAT_MAX_TOKENS` | `2000` | 单次回答长度上限 |
| `MAX_CONTEXT_CHARS` | `12000` | 拼进提示词的参考资料总长度上限 |
| `MAX_HISTORY_ROUNDS` | `5` | 最多携带的历史轮数 |

**检索与分块**

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `CHUNK_SIZE` | `800` | 单块目标**字符数**（不是 token） |
| `CHUNK_OVERLAP` | `150` | 相邻块的重叠字符数 |
| `RETRIEVAL_TOP_K` | `8` | 默认返回片段数 |
| `MAX_RETRIEVAL_TOP_K` | `50` | 调用方传 `topK` 时的上限 |
| `MIN_RELEVANCE_SCORE` | `0.35` | 低于此相似度的片段不放进参考资料 |
| `EMBEDDING_BATCH_SIZE` | `25` | 单批嵌入条数（DashScope 接口上限） |
| `MAX_CHUNKS_PER_DOCUMENT` | `2000` | 单文档最大分块数，防超大文件烧钱 |
| `CHROMA_PATH` | `data/chroma` | 向量库持久化目录，**必须用相对路径** |
| `CHROMA_COLLECTION` | `mindforge_chunks` | 集合名 |

**轮询与服务**

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `ENABLE_POLLER` | `true` | 关掉后不会处理任何文档（调试用） |
| `POLL_INTERVAL_SECONDS` | `5` | 空转休眠秒数，别调太小否则刷爆日志 |
| `CLAIM_LIMIT` | `10` | 单次认领文档数上限 |
| `VECTOR_PURGE_EVERY_N_ROUNDS` | `12` | 每 N 轮清理一次已删除文档的残留向量 |
| `SERVICE_PORT` | `8000` | 服务端口（实际以 uvicorn 启动参数为准） |

---

## 6. 接口文档

### 6.1 通用约定

- **鉴权**：所有 `/internal/**` 请求必须带 `X-Service-Token`，值为 `JAVA_SERVICE_TOKEN`。健康检查不需要。
- **追踪**：可选带 `X-Trace-Id`（日志串联）与 `X-User-Id`（AI 用量归属），本服务不校验。
- **响应信封**：成功与失败都是 `{code, message, data}`，`code === 0` 才算成功。字段名一律 **camelCase**（与 Java 的 record 对齐，不做 snake_case 转换）。
- **SSE 例外**：`/rag/chat/stream` 只能返回 HTTP 200，错误通过 `error` 事件表达。

**错误码**

| code | HTTP | 含义 | 调用方该做什么 |
| :--- | :--- | :--- | :--- |
| `0` | 200 | 成功 | — |
| `40001` | 400 | 参数不合法 / 请求体校验失败 | 修请求 |
| `40101` | 401 | 令牌不对 | 核对 `JAVA_SERVICE_TOKEN` |
| `40301` | 403 | 无权限 | 检查令牌是否被禁用 |
| `40404` | 404 | 路径不存在 | — |
| `42201` | 422 | 文件类型不支持 / 文件读不出文字 / 下载地址过期 | 提示用户换文件或重试 |
| `50000` | 500 | 未预期异常 | 看日志 |
| `50201` | 502 | 模型输出不是合法 JSON、或连不上文件存储 | 重试；持续失败则查提示词 |
| `50202` | 502 | 模型 / 嵌入服务调用失败（网络、额度、鉴权） | 查模型服务与额度 |

### 6.2 接口清单

基础路径 `/internal/ai/v1`（除健康检查外都需要令牌）。

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| POST | `/rag/search` | 知识库语义检索，只返回片段 |
| POST | `/rag/chat` | 检索 + 生成回答（同步） |
| POST | `/rag/chat/stream` | 同上，SSE 流式 |
| POST | `/resumes/parse` | 简历解析为结构化数据 |
| POST | `/resumes/analyze` | 针对岗位给出简历改进建议 |
| POST | `/resumes/match` | 简历与职位匹配分析（五类结论） |
| POST | `/jobs/parse` | 职位 JD 解析为结构化数据 |
| POST | `/interviews/generate` | 根据简历与 JD 出面试题 |
| POST | `/interviews/analyze` | 评估候选人对单道题的回答 |
| POST | `/interviews/report` | 生成整场复盘报告 |
| POST | `/salary/estimate` | 薪酬估算（P50 / P75 / P90） |
| POST | `/tools/url-extract` | 网页正文提取为 Markdown |
| POST | `/calendar/extract-invitation` | 面试邀约抽取为日程候选 |
| GET | `/config/effective` | 当前生效的 AI 配置（供管理端展示） |
| GET | `/status` | AI 能力状态（供首页徽章） |

健康检查（无前缀、免令牌）：

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/health` | 服务存活，适合做容器探针 |
| GET | `/health/java` | 探测 Java 连通性与令牌，**有副作用**，仅手动排查用 |

### 6.3 检索

```http
POST /internal/ai/v1/rag/search
X-Service-Token: <token>
Content-Type: application/json

{
  "userId": "1001",
  "knowledgeBaseId": "1001",
  "query": "Spring Boot 自动装配原理是什么？",
  "topK": 5,
  "enableRerank": true
}
```

```json
{
  "code": 0,
  "message": "成功",
  "data": {
    "results": [
      { "documentId": "12", "chunkIndex": 0, "score": 0.92, "content": "..." }
    ],
    "total": 1,
    "reranked": true
  }
}
```

`reranked` 表示重排**是否真的生效**：没配置 RERANK 模型或调用失败时会降级回向量序，此时为 `false`，调用方可据此决定要不要展示「已精排」标记。

### 6.4 流式问答

```http
POST /internal/ai/v1/rag/chat/stream
Accept: text/event-stream
```

请求体与 `/rag/chat` 相同：`userId`、`knowledgeBaseId`（为空表示通用问答）、`query`、`topK`、`enableRerank`、`promptId`、`history`。

事件序列：

```text
event: sources
data: {"type":"sources","hasContext":true,"sources":[...],"contextTruncated":false}

event: delta
data: {"type":"text","text":"Spring Boot "}

event: delta
data: {"type":"text","text":"通过 @EnableAutoConfiguration..."}

event: done
data: {"type":"answer","hasAnswer":true,"answer":"完整聚合文本"}
```

出问题时是 **先 200 再发 `error` 事件**（代替 `done`），`data` 里带 `code` 与 `message`。**调用方必须处理 `error` 事件**，否则会一直转圈。

两个容易混的字段：

- `hasContext`：有没有检索到片段 —— 用于「参考了 N 个片段」；
- `hasAnswer`：模型有没有给出实质回答 —— 用于「要不要显示答案」。

两者都为 true 是常态；`hasContext=true` + `hasAnswer=false` 也常见：检索到了片段，但片段里没有用户要的信息，模型如实说没有。

### 6.5 面试邀约抽取

```http
POST /internal/ai/v1/calendar/extract-invitation
```

`text` 与 `contentBase64` **二选一**（都给或都不给返回 400）：

```json
{ "userId": "1001", "text": "HR 发来的原话……" }
{ "userId": "1001", "fileType": "PDF", "contentBase64": "JVBERi0xLjQK..." }
```

`fileType` 支持 `PDF` / `DOCX` / `TXT` / `MD` / `PNG` / `JPG`；`contentBase64` 不带 `data:` 前缀。

```json
{
  "code": 0,
  "message": "成功",
  "data": {
    "sourceText": "解析出来的正文……",
    "events": [
      {
        "title": "字节跳动 技术一面",
        "companyName": "字节跳动",
        "positionName": null,
        "eventType": "INTERVIEW_1",
        "eventAt": "2026-10-30T14:00:00",
        "eventAtDerived": false,
        "durationMinutes": 60,
        "eventMode": "ONSITE",
        "location": "上海张江",
        "meetingUrl": null,
        "interviewer": "王琳",
        "note": null,
        "evidence": { "eventType": "面技术一面", "eventAt": "10 月 30 日（周五）下午 2 点" },
        "matchedJobId": null,
        "matchedJobTitle": null
      }
    ],
    "sourceLength": 83,
    "warning": null,
    "truncated": false
  }
}
```

三条契约要点：

1. **`sourceText` 必须回传**。文件输入时 Java 手上没有原文，没它就做不了 `原文.contains(evidence)` 的证据校验，防幻觉链路会整条失效。
2. **`eventAt` 是北京时间、不带时区**（`2026-10-30T14:00:00`）。带 `Z` / `+09:00` 的输入会换算到 UTC+8 再输出 —— ICS 里的 `...Z` 是 UTC，直接当本地时间用会让提醒提前 8 小时发出去。
3. **抽不到就是 `null`**，不填默认值。`eventType` 判断不了轮次就是 `null`；`eventAt` 只有相对说法（「下周找个时间」）时也是 `null`，不会编一个日期。`eventAtDerived=true` 表示这个时间是从相对说法换算来的，前端应标出让用户核对。

不落库、无副作用：返回的是候选，用户在前端确认后才由 Java 写日程表。

---

## 7. 部署

### 7.1 Docker

```bash
docker build -t mindforge-ai-service .
docker run -d --name mindforge-ai-service \
  -p 127.0.0.1:8000:8000 \
  --env-file .env \
  -v $(pwd)/data/chroma:/app/data/chroma \
  mindforge-ai-service
```

镜像基于 `python:3.12-slim`，用 `uv pip install --system` 装依赖，暴露 8000。**`data/chroma` 必须挂载出来**，否则容器重建会丢向量库。

### 7.2 systemd

仓库提供了现成的单元文件 `deploy/mindforge-ai-service.service` 与生产环境变量模板 `deploy/env.production.example`：

```bash
sudo cp deploy/mindforge-ai-service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mindforge-ai-service
journalctl -u mindforge-ai-service -f
```

单元文件里已经固定了几件关键的事：工作目录必须是项目根（`pydantic-settings` 从这里读 `.env`，Chroma 也相对这里持久化）、只监听 `127.0.0.1`、**单 worker**、内存上限 800M。

### 7.3 为什么只能单 worker

Chroma 本地嵌入式模式（SQLite/DuckDB 后端）不支持多进程并发写，多 worker 会写坏数据。另外 2 核 2G 的机器也扛不住多进程。**吞吐不够时先扩内存，再谈 worker。**

---

## 8. 测试

```bash
# 单元测试：不连外部服务、不调模型，几秒跑完
pytest tests/unit -v

# 只跑某个模块
pytest tests/unit/test_invitation_extract.py -v

# 端到端：真实链路，需要 Java + MySQL + Redis + 本服务都在跑
pytest -m e2e -v
```

`pytest` 默认带 `-m 'not e2e'`，所以日常直接 `pytest` 就是跑单元测试。e2e 会真调模型、烧 token、跑十几分钟，且会在库里创建并清理测试数据，**不要在有真实数据的库上跑**。

---

## 9. 常见问题

### 9.1 所有内部接口都返回 401

`JAVA_SERVICE_TOKEN` 与 Java 侧的 `mindforge.internal.service-token` 不一致。注意三点：**大小写、末尾的 `=`、首尾不能有空格**。改完重启本服务（配置只在启动时读一次）。

### 9.2 启动报 Chroma 路径错误，或部署后向量库「丢了」

`CHROMA_PATH` 必须用**相对路径**（默认 `data/chroma`），不要写 `D:\xxx` 这种盘符。代码会把它拼到项目根目录下，这样容器化和 Linux 部署才能无缝迁移。

### 9.3 开了多个 uvicorn worker，向量库损坏

见 [7.3](#73-为什么只能单-worker)。Chroma 本地模式不支持多进程写，请保持单 worker。

### 9.4 简历解析或邀约抽取很慢

先确认 `EXTRACT_ENABLE_THINKING=false`。qwen3 系列默认开「思考」，思考 token 会计入 `completion_tokens` 但**不出现在返回内容里**，等于用户白等：

| 场景 | 思考开 | 思考关 |
| :--- | :--- | :--- |
| 简历解析（2746 字） | 8634 token / 164 秒 | 2301 token / 39 秒 |
| 邀约抽取（含两轮面试的邮件） | 跑满单次超时直接失败（当时 60 秒） | 6.5 秒 |

抽取类任务做的是「照抄原文里有什么」，不需要先推理一遍。给新接口配超时预算时也要按**关思考后**的量算，按开着的算会高估 3~4 倍。

### 9.5 报 42201「文件下载失败（HTTP 403）」

预签名 URL 过期了。Java 签发的下载地址有时效，文件在队列里待久了就会失效 —— 重新入队即可。这不是本服务的问题，日志里会打出具体的状态码和 URL。

### 9.6 报 50201「模型没有按约定返回结构化数据」

模型输出不是合法 JSON（常见于输出被 `max_tokens` 截断，或提示词被改坏）。排查顺序：看该场景的 `max_tokens` 是否够 → 看管理端是否覆盖了提示词版本 → 用 `/internal/ai/v1/config/effective` 确认当前生效的模型。

### 9.7 管理端改了模型 / 提示词，但没生效

两者都是每 60 秒从 Java 拉一次。看启动日志里 `AI 模型配置 : ...` 与 `提示词配置 : ...` 两行，确认当前用的是「管理端」还是「回退 .env / prompts」。拉取失败只会打 WARNING，不影响启动。

---

## 10. 目录结构

```text
mindforge-ai-service/
├── app/
│   ├── main.py                  # FastAPI 入口：生命周期、异常处理与统一响应信封
│   ├── config.py                # 集中配置（pydantic-settings 读 .env）
│   ├── api/
│   │   ├── health.py            # /health 与 /health/java
│   │   └── internal.py          # /internal/ai/v1/** 全部接口
│   ├── core/
│   │   └── security.py          # X-Service-Token 校验
│   ├── clients/
│   │   └── java_client.py       # 反向调用 Java 的 claim / status / chunks / purge
│   ├── services/
│   │   ├── parser.py            # 多格式解析（含扫描件 OCR 降级、截图识别）
│   │   ├── downloader.py        # 预签名 URL 流式下载（校验重定向 + 大小上限）
│   │   ├── splitter.py          # 重叠窗口分块
│   │   ├── embedding.py         # 嵌入模型调用（分批）
│   │   ├── vector_store.py      # Chroma 持久化与检索
│   │   ├── rerank.py            # 交叉重排（未配置/失败降级回向量序）
│   │   ├── pipeline.py          # 文档处理编排（解析→分块→向量化→回写，失败回滚）
│   │   ├── rag.py               # 检索编排（召回放大→精排收窄→阈值过滤）
│   │   ├── chat.py              # 同步与流式问答（含未命中降级）
│   │   ├── llm.py               # 模型调用层（配置变更自动重建 client）
│   │   ├── prompt.py            # 提示词渲染（管理端版本优先 → prompts/*.txt 兜底）
│   │   ├── prompt_store.py      # 管理端提示词缓存
│   │   ├── ai_config.py         # 模型配置解析（管理端优先 → .env 兜底，定时刷新）
│   │   ├── ai_usage.py          # 用量记账（尽力而为，失败不阻塞）
│   │   ├── json_utils.py        # 模型 JSON 输出的矫正与清洗
│   │   ├── resume.py            # 简历解析与定向优化
│   │   ├── job.py               # 职位 JD 解析
│   │   ├── match.py             # 简历-JD 匹配
│   │   ├── interview.py         # 模拟面试出题 / 评分 / 复盘
│   │   ├── salary.py            # 薪酬估算
│   │   ├── invitation.py        # 面试邀约抽取
│   │   └── tools/
│   │       └── web_extractor.py # 网页正文提取（防 SSRF）
│   └── worker/
│       └── poller.py            # 后台轮询认领
├── prompts/                     # 提示词模板（管理端未配置时兜底）
├── data/chroma/                 # 向量库持久化目录（不进版本库）
├── deploy/                      # systemd 单元与生产环境变量模板
├── tests/
│   ├── unit/                    # 单元测试
│   └── e2e/                     # 端到端测试（默认不跑）
├── .env.example                 # 环境变量模板
├── Dockerfile
└── pyproject.toml               # 依赖声明（没有 requirements.txt）
```

---

## 11. 开发约定

- **只做计算**：不直连业务数据库、不判断权限、不写业务状态。需要 Java 加字段或加接口时，先在接口文档里对齐契约再动手。
- **依赖管理**：用 `uv add <pkg>`（会同时更新 `pyproject.toml` 与 `uv.lock`）；**不要**手写 `requirements.txt`。装依赖时显式写版本范围，别追最新大版本。
- **代码里不写 emoji**（注释、日志、报错文案都不行）；`★`、`→`、`⚠` 这类排版符号可以用。
- **提交**：conventional 前缀 + 中文描述，按模块拆小笔；`git push` 由维护者自己做。
- **文档**：仓库里**只有 `README.md` 进版本库**（`.gitignore` 是 `*.md` + `!README.md`），其余 Markdown 是本地资料。
- **新增接口时**：错误要用统一的 `{code, message, data}` 信封（在 `main.py` 里注册对应的异常处理器），字段名用 camelCase，并在本文档 [6.2](#62-接口清单) 补一行。
