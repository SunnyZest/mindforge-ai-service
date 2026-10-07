# MindForge AI Service

MindForge AI 的 Python AI 服务：文档解析、分块、向量化、RAG 检索与对话。

**只给 Java 后端调用**，不对浏览器暴露。

## 与 Java 的分工

```
Spring Boot（Java）          本服务（Python）
─────────────────────       ────────────────────
拥有数据、权限、事务     ←→   只做计算，不连数据库、不判权限
```

这条边界是硬性的 —— 文档 §十七 明确要求「Python 不自行决定用户是否有权限访问资源」。

## 通信方式

**本服务主动轮询 Java**（不是 Java 调本服务）：

```
每 5 秒：
  POST /internal/documents/claim       认领待处理文档（Java 会标成 PARSING）
       ├─ 有任务 → 下载 → 解析 → 分块 → 向量化
       │            PUT  /internal/documents/{id}/status    回写进度
       │            POST /internal/documents/{id}/chunks    写分块
       └─ 无任务 → sleep 5 秒
```

**为什么用轮询而不是 MQ**：Java 在 Windows 本机、本服务将来可能跑在 VM，
NAT 网络下「VM → Windows」方向天然可达，不需要配端口转发。

## 快速开始

```powershell
# 1. 建虚拟环境（uv 会读 .python-version 用 3.12）
uv venv

# 2. 装依赖
uv sync

# 3. 配置
Copy-Item .env.example .env
#    然后编辑 .env，填 JAVA_SERVICE_TOKEN 和 DASHSCOPE_API_KEY

# 4. 启动
uv run uvicorn app.main:app --reload --port 8000
```

## 验证连通性

启动后看日志：

```
✅ 已连上 Java（http://localhost:8080/api），当前待处理文档 0 篇
```

或者手动探测：

```
浏览器打开 http://localhost:8000/health/java
```

**能返回就说明两件事**：Java 可达 + 服务令牌正确。

```
{"status":"ok","javaBaseUrl":"http://localhost:8080/api","claimed":0}
```

如果返回 `{"status":"error", ... "401 ..."}` → `.env` 的 `JAVA_SERVICE_TOKEN`
和 Java 的 `mindforge.internal.service-token` 不一致。

## 与 Java 的接口契约

本服务通过 Java 的 **3 个内部接口**与后端交互。

**所有请求都必须带 `X-Service-Token` 请求头**，值等于 Java 的
`mindforge.internal.service-token`（两边配成完全一致）。

### ① 认领待处理文档

```http
POST /internal/documents/claim?limit=10
```

```json
{
  "code": 0,
  "data": [
    {
      "documentId": "18",
      "knowledgeBaseId": "10",
      "userId": "2",
      "fileName": "spring-boot.txt",
      "fileType": "TXT",
      "fileSize": 5469,
      "downloadUrl": "https://mindforge-ai.oss-cn-beijing.aliyuncs.com/...?Expires=..."
    }
  ]
}
```

**关键点**：

- 这是**认领**不是查询 —— Java 会把返回的文档状态改成 `PARSING`
- **多实例安全**：Java 用 `FOR UPDATE SKIP LOCKED`，不同实例拿到不同的文档
- `data` 为空数组 = 当前没任务（**正常情况，不是错误**）
- `downloadUrl` 是 OSS **预签名地址**（有时效）—— 本服务不需要配 OSS 密钥

### ② 回写处理状态

```http
PUT /internal/documents/{id}/status
```

```json
{
  "parseStatus": "SUCCESS",
  "chunkStatus": "SUCCESS",
  "vectorStatus": "PENDING",
  "errorMessage": null,
  "pageCount": 1
}
```

**字段语义**：

| 字段 | 说明 |
|---|---|
| `parseStatus` / `chunkStatus` / `vectorStatus` | `PENDING` → 处理中 → `SUCCESS` / `FAILED` |
| **传 `null`** | **表示「不修改这个字段」** —— 所以可以分阶段上报 |
| `errorMessage` | **例外**：只要不是 `null` 就写入（传 `""` 用来清空上次的错误）|
| `pageCount` | 页数，`null` 表示不改 |

### ③ 批量写入分块

```http
POST /internal/documents/{id}/chunks
```

```json
{
  "chunks": [
    {
      "chunkIndex": 0,
      "content": "分块正文……",
      "tokenCount": 200,
      "pageNo": 1,
      "vectorId": "chroma-uuid",
      "metadataJson": "{\"section\":\"intro\"}"
    }
  ]
}
```

**关键点**：

- **幂等**：Java 内部先按 `document_id` 物理删、再批量插，
  重复调用不会撞 `uk_doc_chunk_index` 唯一索引
- 返回的 `data` 是写入条数
- `chunks` 不能是空数组，`content` 不能为空

### ④ 查询待清理向量的文档

```http
GET /internal/documents/pending-vector-purge?limit=100
```

```json
{
  "code": 0,
  "data": [
    { "documentId": "26", "knowledgeBaseId": "17", "userId": "2", "fileName": "a.txt" }
  ]
}
```

**为什么需要**：删除文档只做逻辑删除（`deleted = 1`），向量库里的向量不会自动消失 ——
会留下「孤儿向量」，检索时可能召回已被删除的文档内容。
本服务定期拉这个列表，删掉向量后再回写「已清理」。

**为什么是「本服务来拉」而不是「Java 通知本服务」**：本服务将来可能跑在 NAT 后的虚拟机里，
Java 主动调它需要端口转发，而这个方向天然可达。

### ⑤ 标记向量已清除

```http
PUT /internal/documents/{id}/vector-purged
```

**幂等** —— 重复调用不会出错（第二次影响 0 行，按成功处理）。

### 状态流转

```
PENDING ──认领──→ PARSING ──解析成功──→ SUCCESS（parse_status）
                                  ↓
                              CHUNKING ──→ SUCCESS（chunk_status）
                                  ↓
                              EMBEDDING ──→ SUCCESS（vector_status）
                                  ↓
                              三个阶段都 SUCCESS 才「可被检索」
任意阶段出错 ──→ FAILED + error_message（用户可在前端点重试）
```

> ⚠️ **RAG 检索时必须过滤 `deleted = 0` 且 `vector_status = SUCCESS`**，
> 否则会召回还没处理完、或者已被用户删掉的文档内容。

### 错误码

| HTTP | code | 含义 |
|---|---|---|
| 401 | 40101 | 服务令牌无效 —— **两边 token 不一致**（最常见的错）|
| 404 | 40404 | 文档不存在 |
| 400 | 40001 | 参数不合法（比如空分块列表）|
| 409 | 40902 | 状态冲突 |
| 500 | 50000 | 服务端错误 |

## 本服务的接口文档

启动后访问：

```
http://localhost:8000/docs        Swagger UI（FastAPI 自带）
http://localhost:8000/redoc       ReDoc
```

## 目录结构

```
app/
├── main.py                  FastAPI 入口 + 启动/关闭钩子
├── config.py                配置（pydantic-settings 读 .env）
├── api/
│   └── health.py            健康检查
├── core/
│   └── security.py          X-Service-Token 校验
├── clients/
│   └── java_client.py       调 Java 的 claim / status / chunks
├── services/
│   ├── parser.py            文件 → 纯文本
│   ├── splitter.py          文本分块
│   ├── embedding.py         文本 → 向量（DashScope）
│   ├── vector_store.py      向量存储与检索（Chroma）
│   └── pipeline.py          文档处理流水线
└── worker/
    └── poller.py            轮询循环
data/chroma/                 Chroma 向量库持久化目录
```

## 依赖管理

用 [uv](https://docs.astral.sh/uv/)。常用命令：

```powershell
uv add xxx          加依赖（会写进 pyproject.toml 和 uv.lock）
uv sync             按 uv.lock 同步环境
uv run xxx          在项目环境里执行命令，不用手动 activate
```

> `uv.lock` 要提交到 git（锁定依赖版本）；`.venv` 不提交。
