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

    # ---------- 分块 ----------
    # 文档里建议 800/150；原 agent 项目用的是 200/20（偏碎，检索时上下文不够）
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ---------- 本服务 ----------
    service_port: int = 8000


settings = Settings()
