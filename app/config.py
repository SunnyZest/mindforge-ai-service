"""全局配置。

所有配置项都从环境变量 / .env 文件读，不写死在代码里 —— 换环境只改 .env。

优先级（从高到低）：
  1. 环境变量        —— 临时覆盖用，比如 `CHUNK_SIZE=999 uv run ...`
  2. .env 文件
  3. 下面代码里的默认值

⚠️ 大小写不敏感：字段 `java_base_url` 对应 `JAVA_BASE_URL`，两种写法都能识别。
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# app/config.py → parents[1] 就是项目根目录
#
# ⚠️ 必须用绝对路径，不能写 env_file=".env" ——
# 相对路径是相对【当前工作目录】的，从别的目录启动就读不到 .env，
# 配置会静默退回代码里的默认值（token 变成空字符串，所有内部请求 401）。。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",          # .env 里有别的键也不报错
    )

    # ---------- Java 后端 ----------
    # 注意要带 context-path：Java 侧 server.servlet.context-path=/api
    java_base_url: str = "http://localhost:8080/api"

    # 必须和 Java 的 mindforge.internal.service-token 完全一致（包括大小写和末尾的 =）
    java_service_token: str = ""

    # ---------- DashScope（LLM + 嵌入）----------
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model: str = "deepseek-v4-flash"
    embedding_model: str = "text-embedding-v1"

    # ---------- 轮询 ----------
    # 没有任务时的休眠秒数。别设太小 —— 会把 Java 的日志刷爆。
    poll_interval_seconds: int = 5
    # 单次认领的文档数上限（Java 侧还有 claim-limit 兜底）
    claim_limit: int = 10
    # 是否在应用启动时就开启轮询（调试时可先关掉）
    enable_poller: bool = True
    # 每隔多少轮清理一次「已删除文档的向量」。
    # 认领是每 poll_interval_seconds 一轮，清理不需要那么勤 ——
    # 12 轮约 1 分钟，延迟完全可以接受。
    vector_purge_every_n_rounds: int = 12

    # ---------- 分块 ----------
    # 文档里建议 800/150；原 agent 项目用的是 200/20（偏碎，检索时上下文不够）
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ---------- 向量库 ----------
    # Chroma 的持久化目录（相对项目根，代码里会转成绝对路径，不依赖 CWD）
    chroma_path: str = "data/chroma"
    chroma_collection: str = "mindforge_chunks"

    # ---------- 嵌入 ----------
    # DashScope 的 embedding 接口单次最多 25 条，超了会报错，所以要分批
    embedding_batch_size: int = 25
    # 单文档最多切多少块 —— 防御性上限。
    # 超大文件可能切出几万块，把 embedding API 打爆（也烧钱）。
    max_chunks_per_document: int = 2000
    # 检索时返回的片段数
    retrieval_top_k: int = 8
    # 调用方传 topK 时的上限兜底 ——
    # 传太大会把一堆无关片段塞进 prompt，既费钱又降低答案质量
    max_retrieval_top_k: int = 50

    # ---------- 对话 ----------
    # 温度：问答场景要调低，越低越不容易编造。
    # 用模型默认的 1.0 会让它在资料里没有答案时「自由发挥」。
    chat_temperature: float = 0.3
    # 单次回答的长度上限
    chat_max_tokens: int = 2000
    # 拼进 prompt 的参考资料总长度上限（字符）。
    # 超了会被截断 —— 与其让模型收到超长 prompt 报错，不如主动截断并说明。
    max_context_chars: int = 12000
    # 多轮对话最多带几轮历史（太多会挤占参考资料的空间）
    max_history_rounds: int = 5
    # 相似度低于这个值的片段不放进参考资料 ——
    # 宁可不答，也别拿不相关的内容硬凑
    min_relevance_score: float = 0.35

    # ---------- 结构化抽取（简历 / JD 解析）----------
    # ⚠️ 【不能复用 chat_max_tokens】—— 那是给「对话回答」定的 2000。
    #    抽取任务要一次吐一整个 JSON：技能列表 + 每段工作经历的 description
    #    （单条就允许 4000 字）+ 项目 + 教育 + 证书，很容易超过 2000 被截断。
    #    截断的表现是「模型输出不是合法 JSON」（末尾花括号不闭合），
    #    看起来像提示词写坏了，其实只是输出上限太小 —— 排查方向很容易被带偏。
    extract_max_tokens: int = 8000
    # ⚠️ 同理，抽取的【超时】也不能复用对话的 120 秒。
    #    实测：一份 2800 字的简历，模型要吐几千 token，跑满 120 秒是常态
    #    （踩过 —— Java 侧读超时也是 120 秒，两边赛跑，Java 先掐断连接，
    #      客户端拿到的是「响应读不出来」而不是「超时」，完全看不出真因）。
    #
    # ★★ 超时预算必须把【重试】算进去 —— 2026-10-10 踩过一次：
    #    OpenAI SDK 默认会自动重试（默认 max_retries=2）。一次 300 秒的超时
    #    不会直接失败，而是【再试一遍】，所以实际最坏耗时是
    #        单次超时 × (1 + 重试次数)
    #    而不是单次超时。实测日志：第一次尝试 10:27:42 超时 → SDK 重试
    #    → 10:30:43 成功，全程约 8 分钟，而 Java 的 360 秒早就到了 ——
    #    用户看到「AI 服务超时」，但 Python 其实成功了。
    #
    #    ⚠️ 所以规则是：Java 的 read-timeout > 下面这两个值的乘积
    #       extract_timeout_seconds × (1 + llm_max_retries)
    #       现在 = 300 × 2 = 600 秒，Java 配的是 660 秒。
    extract_timeout_seconds: int = 300

    # 调用模型的自动重试次数。
    # ⚠️ 显式写出来而不是吃 SDK 默认值（默认 2）—— 默认值会让最坏耗时变成 3 倍，
    #    和 Java 的读超时对不上（见上面那段）。1 次是折中：
    #    连接抖动这类「重试一下就好」的错误能救回来，
    #    又不会把一个已经跑了 5 分钟的超时再拖 5 分钟。
    llm_max_retries: int = 1

    # ---------- 本服务 ----------
    service_port: int = 8000


settings = Settings()
