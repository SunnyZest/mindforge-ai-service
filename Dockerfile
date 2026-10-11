# ==============================================================================
# MindForge AI Service Python AI 微服务 Dockerfile
# Python 3.12 + FastAPI + ChromaDB + Uvicorn
# ==============================================================================

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 安装基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖定义
COPY pyproject.toml uv.lock* ./

# 利用 uv 加速全局依赖安装
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r pyproject.toml

# 复制代码与 prompts 模板
COPY app/ app/
COPY prompts/ prompts/

# 预建持久化数据目录
RUN mkdir -p /app/data/chroma

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
