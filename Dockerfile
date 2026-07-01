FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update -qq && apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY dce_gpu_faceswap.py /app/dce_gpu_faceswap.py
COPY prefetch_models.py /app/prefetch_models.py

# Preinstall heavy model artifacts during image build, not during first request.
RUN mkdir -p /models /tmp/dce-video-gpu
ENV DCE_MODEL_DIR=/models \
    DCE_INSIGHTFACE_ROOT=/models/insightface \
    DCE_WORK_DIR=/tmp/dce-video-gpu
RUN python3 /app/prefetch_models.py

CMD ["python3", "-u", "/app/handler.py"]
