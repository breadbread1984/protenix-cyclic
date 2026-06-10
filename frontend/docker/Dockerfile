FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUTLASS_PATH=/opt/cutlass

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install protenix --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages

# Install protenix and frontend dependencies
RUN python3 -m pip install gradio gradio_rangeslider fastapi uvicorn --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages

# Copy frontend code
COPY frontend/main.py /app/frontend/main.py
COPY frontend/runner.py /app/frontend/runner.py

# Create output directory
RUN mkdir -p /root/protenix_output

WORKDIR /app/frontend
