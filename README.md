# MindForge AI Service（Python AI 计算微服务）

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/ChromaDB-0.5+-FF6F61?style=flat-square" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/DashScope-Alibaba%20Cloud-FF6A00?style=flat-square" alt="DashScope" />
  <img src="https://img.shields.io/badge/Trafilatura-2.3-2E7D32?style=flat-square" alt="Trafilatura" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License" />
</p>

> **MindForge AI Service** 是 MindForge AI 系统的核心 AI 计算引擎与文本向量化中枢。  
> 本服务采用 **FastAPI 异步架构** 与纯粹的 **无状态计算节点设计**，专注于异构多源文档智能解析、工业级网页正文提取与清洗、向量嵌入计算（Embedding）、Chroma 向量检索、SSE 逐字流式 RAG 问答推流及大模型推理。
>
> 🔒 **服务安全边界**：本服务定位为**内部计算中枢**，仅接受 Java 业务后端的调用（携带 `X-Service-Token`），**不直接向浏览器公网开放**。

---

## 目录

- [一、 架构定位与设计哲学](#一-架构定位与设计哲学)
- [二、 核心功能特性](#二-核心功能特性)
- [三、 目录结构与模块说明](#三-目录结构与模块说明)
- [四、 本地环境搭建与启动](#四-本地环境搭建与启动)
- [五、 与 Java 主后端通信契约](#五-与-java-主后端通信契约)
  - [1. 异步轮询认领流水线 (Pull 模式)](#1-异步轮询认领流水线-pull-模式)
  - [2. 原生网页正文抽取工具接口](#2-原生网页正文抽取工具接口)
  - [3. 同步语义检索接口](#3-同步语义检索接口)
  - [4. RAG 知识问答与 SSE 流式推流](#4-rag-知识问答与-sse-流式推流)
- [六、 自动化测试与质量保障](#六-自动化测试与质量保障)
- [七、 避坑指南与生产调优](#七-避坑指南与生产调优)
- [八、 未来演进路线图](#八-未来演进路线图)

---

## 一、 架构定位与设计哲学

### 1. 硬性职责边界（数据在 Java，计算在 Python）
```
┌─────────────────────────────────┐           ┌─────────────────────────────────┐
│     Java 业务后端 (Spring Boot)   │           │   Python AI 微服务 (FastAPI)     │
├─────────────────────────────────┤           ├─────────────────────────────────┤
│ • 拥有 MySQL 直连权与全局事务   │           │ • 纯粹无状态，不连数据库/不判权 │
│ • 负责用户鉴权与 RBAC 权限隔离  │  REST/SSE │ • 异构文档清洗与网页正文抓取    │
│ • 生成 OSS / MinIO 临时预签名   │ ◄───────► │ • 分块与 DashScope 向量嵌入     │
│ • 会话上下文与业务状态机流转    │           │ • Chroma 向量索引与相似度检索   │
└─────────────────────────────────┘           │ • SSE 首字 0.45s 逐字流式推流   │
                                              └─────────────────────────────────┘
```
- **绝不越权**：Python 侧不直连业务数据库，不判断用户是否有权限访问某个知识库。所有权限由 Java 在调用前校验完毕。
- **解耦存储**：文档解析时，Python 通过 Java 下发的有时效 **OSS/MinIO 预签名只读 URL** 下载二进制流，无需在 Python 端配置云存储 AccessKey。

### 2. 去中心化异步流水线（主动轮询认领）
- 文档处理是长耗时密集任务，本服务内置后台 Worker（`poller.py`），每 5 秒主动调用 Java 内部接口 `POST /internal/documents/claim`。
- Java 端利用数据库 `FOR UPDATE SKIP LOCKED` 悲观锁保证多节点安全并发；Python 主动拉取任务（Pull 模式），天然防御 NAT 穿透阻断与系统过载背压。

---

## 二、 核心功能特性

### 1. 异构文档归一化解析引擎
无需笨重的 LangChain 依赖，使用轻量原生库统一产出带结构标记的 Markdown 文本：
- **TXT / Markdown**：智能探测 UTF-8 / GBK 编码无损解码；
- **PDF**：基于 `pdfplumber` 高保真提取文字与排版，提取失败时自动使用 `pypdf` 容错兜底；
- **DOCX**：基于 `python-docx` 提取段落与表格，标题保留 `#` 结构，表格统一转为 Markdown 表格；
- **HTML / 网页文章**：集成工业级 `trafilatura` 原生抽取引擎，毫秒级智能剔除广告、导航与页脚，提取高纯度正文标题与 Markdown。

### 2. 智能重叠分块与向量化管道
- **滑动窗口重叠切分**：按设定 `chunk_size`（如 500 token）与 `chunk_overlap`（如 50 token）保留上下文连续性；
- **DashScope 阿里云百炼嵌入**：对接 `text-embedding-v1` 模型生成 1536 维高质量向量；
- **ChromaDB 本地向量持久化**：以 `knowledge_base_id` 为安全集合隔离，支持余弦相似度（Cosine Distance）快速匹配。

### 3. 首字 0.45s 逐字 SSE 流式推流与优雅降级
- 支持标准的 `text/event-stream` SSE 传输协议；
- **零门槛与未命中降级**：当用户未选择知识库或向量检索未能召回高置信度片段时，系统自动优雅降级为通用大模型解答，并附带明确的免责标识，坚决拒绝生硬报错。

### 4. 脏向量孤儿数据闭环清理
- 定时拉取 Java 逻辑删除（`deleted=1`）的文档列表，同步剔除 Chroma 中的历史向量分块，杜绝废弃文档在后续检索中被误召回。

---

## 三、 目录结构与模块说明

```text
mindforge-ai-service/
├── app/
│   ├── main.py                  # FastAPI 应用入口、生命周期钩子（启动轮询器）
│   ├── config.py                # 集中化配置管理（基于 pydantic-settings 读取 .env）
│   ├── api/
│   │   ├── health.py            # 健康自检接口（/health 与 /health/java 连通性测试）
│   │   └── internal.py          # /internal/ai/v1/** 供 Java 调用的同步接口
│   ├── core/
│   │   └── security.py          # X-Service-Token 服务间安全通信校验
│   ├── clients/
│   │   └── java_client.py       # 封装反向请求 Java 的 claim/status/chunks/purge 客户端
│   ├── services/
│   │   ├── parser.py            # 本地上传多格式文件（PDF/DOCX/TXT）解析器
│   │   ├── splitter.py          # 结构化文本分块器（带重叠窗口与标点断句）
│   │   ├── embedding.py         # DashScope 向量嵌入接口封装
│   │   ├── vector_store.py      # ChromaDB 本地持久化与语义检索
│   │   ├── pipeline.py          # 文档处理全流程编排（解析➔分块➔向量化➔回写）
│   │   ├── rag.py               # RAG 知识检索与 Prompt 模板组装
│   │   ├── llm.py               # DashScope Qwen 大模型交互层
│   │   ├── chat.py              # 同步与 SSE 流式 RAG 对话逻辑
│   │   └── tools/
│   │       └── web_extractor.py # 基于 trafilatura 的工业级网页正文提取工具
│   └── worker/
│       └── poller.py            # 后台异步任务轮询认领守护线程
├── data/
│   └── chroma/                  # ChromaDB 向量数据持久化目录（Git 忽略）
├── tests/
│   └── unit/
│       ├── test_web_extractor.py# 网页提取器单元测试
│       ├── test_splitter.py     # 分块逻辑测试
│       └── ...                  # 覆盖核心计算逻辑的单元测试集
├── .env.example                 # 环境变量配置模板
├── pyproject.toml               # uv / pip 项目元数据与依赖定义
└── requirements.txt             # 生产环境依赖版本锁定表
```

---

## 四、 本地环境搭建与启动

### 1. 环境准备
- **Python**：`3.12` 及以上
- 包管理器推荐使用现代极速工具 **[uv](https://docs.astral.sh/uv/)**（亦可使用原生 `pip`）

### 2. 安装项目依赖

```bash
# 方式 A：使用 uv（推荐，秒级安装）
uv venv --python 3.12
uv sync

# 方式 B：使用原生 venv (Windows)
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 方式 C：使用原生 venv (Linux / macOS)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 配置环境变量
复制根目录模板并创建 `.env`：
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

核心配置项详解（`.env`）：
```env
# Java 后端地址（必须包含 context-path /api）
JAVA_BASE_URL=http://localhost:8080/api

# 内部双向通讯密钥（必须与 Java 端 mindforge.internal.service-token 完全一致）
JAVA_SERVICE_TOKEN=5oiQ55Ge56Wl5Zac5qyi5a2Z5Y2a5paH

# 阿里云百炼大模型密钥（必须配置）
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CHAT_MODEL=qwen-plus
EMBEDDING_MODEL=text-embedding-v1

# Chroma 向量数据库持久化目录（推荐使用相对路径）
CHROMA_PERSIST_DIRECTORY=data/chroma

# 启用后台文档认领轮询器
ENABLE_POLLER=true

# 服务端口
SERVICE_PORT=8000
```

### 4. 启动服务

```bash
# 使用 uv 启动
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 或在虚拟环境下直接启动
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 验证健康状态与连通性
- **查看 Swagger API 文档**：浏览器访问 `http://localhost:8000/docs`
- **与 Java 后端连通性自检**：浏览器访问 `http://localhost:8000/health/java`  
  若返回 `{"status":"ok","claimed":0}` 则说明与 Java 业务后端成功握手。

---

## 五、 与 Java 主后端通信契约

所有调用接口（无论 Python 调 Java 还是 Java 调 Python）请求头均必须携带：  
`X-Service-Token: <配置的通讯密钥>`

### 1. 异步轮询认领流水线 (Pull 模式)
Python 后台 Poller 驱动文档流转：
```
1. POST /internal/documents/claim?limit=10 ──► Java 认领并锁定未处理文档
2. Python 下载 OSS 预签名流 ──► 解析 ──► 分块 ──► 写入向量库
3. PUT  /internal/documents/{id}/status    ──► 分阶段上报进度 (PARSING ➔ CHUNKING ➔ SUCCESS)
4. POST /internal/documents/{id}/chunks    ──► 批量持久化分块元数据到 MySQL
```

### 2. 原生网页正文抽取工具接口
- **请求**：`POST /internal/ai/v1/tools/url-extract`
  ```json
  {
    "url": "https://juejin.cn/post/728192837482"
  }
  ```
- **响应**：
  ```json
  {
    "code": 0,
    "message": "提取成功",
    "data": {
      "url": "https://juejin.cn/post/728192837482",
      "title": "深入剖析 Java 垃圾回收器与内存分配策略",
      "content": "# 深入剖析 Java 垃圾回收器...\n\n在 JVM 体系中...",
      "wordCount": 3840
    }
  }
  ```

### 3. 同步语义检索接口
- **请求**：`POST /internal/ai/v1/rag/search`
  ```json
  {
    "knowledgeBaseId": "1001",
    "query": "Spring Boot 自动装配原理是什么？",
    "topK": 5
  }
  ```

### 4. RAG 知识问答与 SSE 流式推流
- **接口**：`POST /internal/ai/v1/rag/chat/stream`
- **事件拓扑**：
  ```text
  event: sources
  data: {"type":"sources","hasContext":true,"sources":[{"documentId":"12","chunkIndex":0,"score":0.92,"content":"..."}]}

  event: delta
  data: {"type":"text","text":"Spring Boot "}

  event: delta
  data: {"type":"text","text":"通过 @EnableAutoConfiguration..."}

  event: done
  data: {"type":"answer","hasAnswer":true,"answer":"完整聚合文本..."}
  ```

---

## 六、 自动化测试与质量保障

项目配备了严谨的单元测试套件，全面覆盖文本抽取、分块边界与 RAG 召回：

```bash
# 运行全部单元测试
pytest tests/unit -v

# 仅测试网页正文抽取工具
pytest tests/unit/test_web_extractor.py -v
```

---

## 七、 避坑指南与生产调优

1. **ChromaDB 跨平台持久化路径**：
   - 务必在 `.env` 中使用相对路径 `data/chroma`，严禁写入诸如 `D:\xxx` 的 Windows 盘符，以保证 Docker 容器化和 Linux 服务器部署无缝迁移。
2. **多进程并发与 Chroma 锁冲突**：
   - Chroma 本地嵌入式模式（DuckDB/SQLite 后端）不支持多进程并发写入。生产环境若开启多 Uvicorn Worker，建议保持单 Poller 线程认领，检索支持只读并发。
3. **网页提取防反爬超时**：
   - `web_extractor` 默认设置了 10 秒超时与标准浏览器 User-Agent 头；遇到极少数强反爬站点（如 Cloudflare 验证盾）时，会自动抛出规整的 400 提示错误，不会导致主进程挂起。

---

## 八、 未来演进路线图

- [ ] **交叉重排（Cross-Encoder Rerank）**：接入 BGE-Reranker 模型，将向量初筛 Top-20 二次精排至 Top-5，进一步提升 RAG 回答精准度；
- [ ] **多模态与 OCR 扩展**：引入 PaddleOCR 支持图片型扫描版 PDF 简历精准识读；
- [ ] **轻量实时语音接入（ASR/TTS）**：接入 SenseVoice / Whisper 流式语音识别与 CosyVoice 语音合成，支撑拟真口头语音面试；
- [ ] **本地私有化大模型直连**：增加 Ollama / vLLM 兼容适配器，支持 DeepSeek-R1 / Qwen 本地离线部署。
