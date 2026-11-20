# 單檔 Dockerfile，無 useradd/groupadd
ARG CUDA_BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
FROM ${CUDA_BASE_IMAGE}

# 基本工具
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git wget curl vim htop tmux ca-certificates build-essential \
    libgl1 libglib2.0-0 libjpeg-dev zlib1g-dev libpng-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 你需要的 Python 套件
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

WORKDIR /workspaces
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 建立與 Ubuntu 主機相同 UID/GID 的使用者
RUN groupadd -g 1000 yhlin \
    && useradd -m -u 1000 -g 1000 -s /bin/bash yhlin

# 設定家目錄
ENV HOME=/home/yhlin

# 建立 VS Code Server 所需目錄
RUN mkdir -p /home/yhlin/.vscode-server \
    && chown -R 1000:1000 /home/yhlin

WORKDIR /workspaces/edge-diffusion