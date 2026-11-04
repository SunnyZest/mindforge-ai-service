# MindForge AI Service

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/FastAPI-0.142+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/ChromaDB-1.5+-FF6F61?style=flat-square" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/DashScope-OpenAI%20Compatible-FF6A00?style=flat-square" alt="DashScope" />
</p>

MindForge AI 的 AI 计算服务。它把用户上传的资料变成可检索的知识库，并基于知识库回答问题；也负责简历、职位、面试邀约这些场景的结构化抽取。

服务本身不连数据库、不判断权限、不持有对象存储密钥 —— 这些都在 Java 后端。它只接受 Java 的调用，用 `X-Service-Token` 鉴权，不直接对浏览器开放。

```text
Java 后端 ──REST/SSE──► 本服务 ──► DashScope（对话 / 嵌入 / 重排）
                            └─────► ChromaDB（向量库，本地磁盘）
```

---

## 快速开始

需要 Python 3.12+ 和一个能连上的 [Java 后端](https://github.com/SunnyZest/mindforge-ai)。

```bash
uv venv --python 3.12
uv sync
cp .env.example .env
```

在 `.env` 里填三项：

```env
JAVA_BASE_URL=http://localhost:8080/api      # 必须带 context-path /api
JAVA_SERVICE_TOKEN=<与 Java 侧 mindforge.internal.service-token 一致>
DASHSCOPE_API_KEY=sk-...
```

启动：

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

确认它活着：

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"mindforge-ai-service","pollerEnabled":true,"pollIntervalSeconds":5}
```

打开 <http://localhost:8000/docs> 可以看到全部接口的 Swagger 文档。

---

## 相关仓库

本服务是 MindForge AI 的一部分，单独跑起来只能验证「服务活着」，真正的业务链路由另外两个仓库驱动：

| 仓库 | 说明 |
| :--- | :--- |
| [SunnyZest/mindforge-ai](https://github.com/SunnyZest/mindforge-ai) | Java 后端：用户与权限、业务数据、对象存储签名。本服务的所有调用方 |
| [SunnyZest/mindforge-web](https://github.com/SunnyZest/mindforge-web) | 前端：用户端与管理端 |

---

## 用法

所有接口都在 `/internal/ai/v1` 下，都要带 `X-Service-Token`。响应统一是 `{code, message, data}`，`code` 为 `0` 才算成功。

### 基于知识库问答（流式）

```bash
curl -N http://localhost:8000/internal/ai/v1/rag/chat/stream \
  -H "X-Service-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"userId":"1001","knowledgeBaseId":"1001","query":"Spring Boot 自动装配原理是什么？"}'
```

```text
event: sources
data: {"type":"sources","hasContext":true,"sources":[{"documentId":"12","chunkIndex":0,"score":0.92,"content":"..."}]}

event: delta
data: {"type":"text","text":"Spring Boot "}

event: delta
data: {"type":"text","text":"通过 @EnableAutoConfiguration..."}

event: done
data: {"type":"answer","hasAnswer":true,"answer":"完整聚合文本"}
```

`hasContext` 表示有没有检索到片段，`hasAnswer` 表示模型有没有给出实质回答 —— 两者都为 true 是常态，`hasContext` 为 true 而 `hasAnswer` 为 false 也常见：找到了片段，但里面没有用户要的信息。

流式接口**只能返回 200**，出错时发一个 `error` 事件代替 `done`：

```text
event: error
data: {"type":"error","code":50202,"message":"AI 回答中断：..."}
```

### 只取检索片段

```bash
curl http://localhost:8000/internal/ai/v1/rag/search \
  -H "X-Service-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"userId":"1001","knowledgeBaseId":"1001","query":"分库分表","topK":5,"enableRerank":true}'
```

```json
{
  "code": 0,
  "data": {
    "results": [{"documentId": "12", "chunkIndex": 0, "score": 0.92, "content": "..."}],
    "total": 1,
    "reranked": true
  }
}
```

`reranked` 表示重排是否真的生效。没配置 RERANK 模型或调用失败时会降级回向量序，此时为 `false`。

### 解析简历

文件由 Java 上传到对象存储，再把预签名地址交过来：

```bash
curl http://localhost:8000/internal/ai/v1/resumes/parse \
  -H "X-Service-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"userId":"1001","resumeId":"88","fileUrl":"https://oss.../cv.pdf","fileType":"PDF","targetPosition":"Java 后端"}'
```

返回结构化简历（summary、教育、工作、项目、技能），以及 `rawText` —— 提取出来的原文要回传给 Java 存库，后续做匹配和优化时不用重新下载解析。这是同步接口，一份简历大约几十秒。

### 抽取面试邀约

三种输入：粘贴的文本、PDF/DOCX、PNG/JPG 截图。

```bash
curl http://localhost:8000/internal/ai/v1/calendar/extract-invitation \
  -H "X-Service-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"userId":"1001","text":"10 月 30 日（周五）下午 2 点来我们上海张江的办公室面技术一面，大概 1 小时"}'
```

```json
{
  "code": 0,
  "data": {
    "sourceText": "10 月 30 日（周五）下午 2 点来我们上海张江的办公室面技术一面，大概 1 小时",
    "events": [{
      "title": "字节跳动 技术一面",
      "companyName": "字节跳动",
      "eventType": "INTERVIEW_1",
      "eventAt": "2026-10-30T14:00:00",
      "eventAtDerived": false,
      "durationMinutes": 60,
      "eventMode": "ONSITE",
      "location": "上海张江的办公室",
      "evidence": {"eventType": "面技术一面", "eventAt": "10 月 30 日（周五）下午 2 点"}
    }],
    "sourceLength": 36,
    "warning": null
  }
}
```

这个接口的设计原则是**只抽不推**：原文没写轮次，`eventType` 就是 `null`；只写「下周找个时间」，`eventAt` 也是 `null`，不会替你编一个日期。每个字段都带 `evidence`（原文片段），调用方拿它做子串校验就能过滤掉模型编的内容。

响应里的 `sourceText` 是解析出来的正文，校验 evidence 要用它 —— 文件输入时调用方手上没有原文，别把它丢掉。

`eventAt` 是北京时间、不带时区。带 `Z` 或 `+09:00` 的输入会先换算到 UTC+8 再输出。

---

## 工作原理

**文档入库是异步的**，由本服务主动去 Java 领任务：

```text
每 5 秒 ──► POST /internal/documents/claim    Java 用 FOR UPDATE SKIP LOCKED 认领
         ◄── 待处理文档（含预签名下载地址）
下载 ──► 解析 ──► 分块 ──► 嵌入 ──► 写入 Chroma
         ──► PUT  /internal/documents/{id}/status    上报 PARSING / CHUNKING / SUCCESS
         ──► POST /internal/documents/{id}/chunks    回写分块元数据
```

拉取而不是推送：不受 NAT 限制，也天然形成背压。

**问答是同步的**：检索（向量召回 → 可选精排 → 阈值过滤）→ 拼提示词 → 生成 → 返回或流式推流。没有检索到可用片段时降级为通用问答，并在 `hasContext` 里如实标记。

向量都存在同一个 Chroma collection 里，用 `knowledge_base_id` 元数据过滤实现知识库隔离。

---

## 配置

模型参数和提示词支持两层来源：**管理端配置优先**，拉不到才用 `.env` 和 `prompts/*.txt`。两者每 60 秒刷新一次，启动日志里会打印当前生效的是哪一套。

常用配置项：

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `JAVA_BASE_URL` | `http://localhost:8080/api` | Java 后端地址，必须带 `/api` |
| `JAVA_SERVICE_TOKEN` | — | 服务间令牌，两边必须完全一致 |
| `DASHSCOPE_API_KEY` | — | 百炼密钥，管理端配了模型时可留空 |
| `CHAT_MODEL` | `deepseek-v4-flash` | 对话模型 |
| `EMBEDDING_MODEL` | `text-embedding-v1` | 嵌入模型 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | 分块大小与重叠，单位是字符 |
| `RETRIEVAL_TOP_K` | `8` | 默认召回片段数 |
| `MIN_RELEVANCE_SCORE` | `0.35` | 低于此相似度的片段不放进上下文 |
| `CHROMA_PATH` | `data/chroma` | 向量库目录，用相对路径 |
| `ENABLE_POLLER` | `true` | 关掉后不处理任何文档 |
| `EXTRACT_ENABLE_THINKING` | `false` | 抽取类场景是否让模型先思考，见「故障排查」 |

完整清单（含超时、重试、批量大小等）见 [`.env.example`](.env.example)。

---

## 接口一览

| 分组 | 接口 |
| :--- | :--- |
| 知识库 | `POST /rag/search` · `POST /rag/chat` · `POST /rag/chat/stream` |
| 简历与职位 | `POST /resumes/parse` · `POST /resumes/analyze` · `POST /resumes/match` · `POST /jobs/parse` |
| 面试 | `POST /interviews/generate` · `POST /interviews/analyze` · `POST /interviews/report` |
| 其他 | `POST /salary/estimate` · `POST /tools/url-extract` · `POST /calendar/extract-invitation` |
| 运维 | `GET /config/effective` · `GET /status` |

路径前缀 `/internal/ai/v1`。字段名一律 camelCase，直接对齐 Java 的 record。

出错时按 HTTP 状态码区分：`40001` 参数不对、`42201` 文件读不出来（含下载地址过期）、`50201` 模型输出不是合法 JSON、`50202` 模型或嵌入服务调不通。

---

## 部署

Docker：

```bash
docker build -t mindforge-ai-service .
docker run -d --name mindforge-ai-service \
  -p 127.0.0.1:8000:8000 \
  --env-file .env \
  -v $(pwd)/data/chroma:/app/data/chroma \
  mindforge-ai-service
```

systemd：仓库里有现成的 `deploy/mindforge-ai-service.service` 与 `deploy/env.production.example`。

```bash
sudo cp deploy/mindforge-ai-service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mindforge-ai-service
journalctl -u mindforge-ai-service -f
```

两个要点：`data/chroma` 必须挂载或持久化，否则重建容器会丢向量库；**只能单 worker** —— Chroma 本地模式不支持多进程写。

---

## 开发

```bash
pytest tests/unit -v          # 单元测试，不连外部服务
pytest -m e2e -v              # 端到端，需要 Java + MySQL + Redis 都在跑
```

日常直接 `pytest` 就是跑单元测试（配置里默认 `-m 'not e2e'`）。e2e 会真调模型、烧 token，别在有真实数据的库上跑。

```text
app/
├── main.py          FastAPI 入口、异常处理与统一响应信封
├── config.py        配置（pydantic-settings）
├── api/             health.py、internal.py
├── clients/         java_client.py（反向调用 Java）
├── services/        parser / splitter / embedding / vector_store / rerank
│                    rag / chat / llm / prompt / ai_config / ai_usage
│                    resume / job / match / interview / salary / invitation
└── worker/          poller.py（后台认领）
prompts/             提示词模板
deploy/              systemd 与生产环境变量模板
tests/               unit / e2e
```

---

## 故障排查

**401 全部接口都失败** —— `JAVA_SERVICE_TOKEN` 和 Java 侧不一致。注意大小写、末尾的 `=`、以及不能有首尾空格。配置只在启动时读一次，改完要重启。

**向量库「丢了」或部署后报路径错误** —— `CHROMA_PATH` 要用相对路径（默认 `data/chroma`），别写 `D:\xxx` 这种盘符。

**简历解析或邀约抽取很慢** —— 先确认 `EXTRACT_ENABLE_THINKING=false`。qwen3 系列默认开着「思考」，思考 token 计入 `completion_tokens` 但不出现在返回内容里，等于让用户白等：同一份简历开着是 164 秒 / 8600 token，关掉是 39 秒 / 2300 token，抽出来的字段一个没少。给新接口配超时预算时也按关掉之后的量算。

**42201 文件下载失败** —— 预签名 URL 过期了，重新入队即可。

**50201 模型没有按约定返回结构化数据** —— 输出不是合法 JSON，通常是 `max_tokens` 不够把 JSON 截断了，或者管理端覆盖的提示词版本有问题。用 `GET /config/effective` 确认当前生效的模型与提示词。

---

## 贡献

依赖用 `uv add` 管理（没有 `requirements.txt`），提交前跑一遍 `pytest tests/unit`。新增接口时请复用统一的 `{code, message, data}` 信封，字段名用 camelCase，并在本文档的接口一览里补一行。

仓库里只有 `README.md` 进版本库，其余 Markdown 是本地资料。
