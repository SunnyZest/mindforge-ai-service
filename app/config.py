"""全局配置。

所有配置项都从环境变量 / .env 文件读，不写死在代码里 —— 换环境只改 .env。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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

    # ---------- 本服务 ----------
    service_port: int = 8000


settings = Settings()
