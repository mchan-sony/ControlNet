FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    vim \
    wget \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    python3 \
    python3-pip \
    libmpich-dev \
    libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt requirements.txt

# install deps
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r requirements.txt
RUN rm requirements.txt

# RUN git config --global user.name "Matthew Chan" \
#     && git config --global user.email "matthew.a.chan@sony.com" \
#     && git lfs install

RUN wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt