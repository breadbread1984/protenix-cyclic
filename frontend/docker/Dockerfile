FROM vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:2.7.1-cu12.6.3-py3.11-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUTLASS_PATH=/opt/cutlass

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        g++ \
        gcc \
        libc6-dev \
        make \
        hmmer \
        kalign \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CUTLASS
RUN git clone -b v3.5.1 https://github.com/NVIDIA/cutlass.git /opt/cutlass

WORKDIR /app

# Install protenix and frontend dependencies
RUN pip3 install --no-cache-dir \
        protenix \
        gradio>=4.0.0 \
        gradio_rangeslider \
        fastapi>=0.100.0 \
        uvicorn>=0.20.0 \
    -i https://pypi.org/simple

# Copy frontend code
COPY frontend/main.py /app/frontend/main.py
COPY frontend/runner.py /app/frontend/runner.py

# Create output directory
RUN mkdir -p /root/protenix_output

WORKDIR /app/frontend



