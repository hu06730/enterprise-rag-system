# RAG 知识库服务镜像
FROM python:3.13-slim

# PYTHONUNBUFFERED=1  : 日志立刻输出（否则 docker logs 要等缓冲满才看得到）
# PYTHONDONTWRITEBYTECODE : 不生成 .pyc，镜像更干净
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE : 强制离线加载模型，避免启动时联网校验失败
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# 先只复制"依赖声明 + 源码"，再装依赖。
# Docker 是按层缓存的：只要这几个文件没变，改业务代码不会触发重新装依赖。
# ⚠️ 必须复制 README.md —— pyproject.toml 里 readme = "README.md"，缺了构建会失败。
COPY pyproject.toml README.md requirements.txt ./
COPY ragkb ./ragkb

# 分两步装：先装依赖（体积大、变化少），再装本项目（--no-deps 避免重复解析依赖树）
RUN pip install -r requirements.txt && pip install --no-deps .

EXPOSE 8000

# --host 0.0.0.0 必须写：容器内监听 127.0.0.1 的话，宿主机转发进来的流量收不到
CMD ["uvicorn", "ragkb.app:app", "--host", "0.0.0.0", "--port", "8000"]
