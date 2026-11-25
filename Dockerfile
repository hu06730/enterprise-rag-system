# RAG 知识库服务镜像
FROM python:3.13-slim

# PYTHONUNBUFFERED=1  : 日志立刻输出（否则 docker logs 要等缓冲满才看得到）
# PYTHONDONTWRITEBYTECODE : 不生成 .pyc，镜像更干净
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE : 强制离线加载模型，避免启动时联网校验失败
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# 先只复制"依赖声明 + 源码"，再装依赖。
# Docker 是按层缓存的：只要这几个文件没变，改业务代码不会触发重新装依赖。
# ⚠️ 必须复制 README.md —— pyproject.toml 里 readme = "README.md"，缺了构建会失败。
# 只先复制"依赖声明"，不复制源码 —— 这样改代码不会让依赖层失效。
COPY pyproject.toml README.md requirements.txt ./

# CPU 版 torch：Linux 下 pip 默认拉 CUDA 版（约 2.5GB nvidia-* 依赖），我们只用 CPU。
# --mount=type=cache 把 pip 下载缓存跨构建保留，第二次构建直接命中本地缓存。
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu torch

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 源码放最后：改代码只会重跑下面两层（秒级），不再重下依赖
COPY ragkb ./ragkb
RUN pip install --no-deps .

EXPOSE 8000

# --host 0.0.0.0 必须写：容器内监听 127.0.0.1 的话，宿主机转发进来的流量收不到
CMD ["uvicorn", "ragkb.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------
# 可选优化（下次重建时启用，能把依赖安装从 ~18 分钟降到 ~1 分钟）：
#   1. 删掉上面的 PIP_NO_CACHE_DIR=1
#   2. 给两个 pip RUN 加缓存挂载：
#        RUN --mount=type=cache,target=/root/.cache/pip pip install ...
#   原理：pip 下载的 wheel 会跨构建复用，不再每次联网重下。
# ---------------------------------------------------------------
