FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

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
    libsm6 \
    libxext6 \
    libxrender-dev \
    git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

## copy the repo
COPY ./ /workspace/controlnet/
WORKDIR /workspace/controlnet/

# install deps
RUN conda init bash \
    && . ~/.bashrc \
    && conda create -n controlnet \
    && python -m pip install --upgrade pip  \
    && conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia \
    && pip install -q -y diffusers transformers accelerate opencv-python numpy

RUN git config --global user.name "Matthew Chan" \
    && git config --global user.email "matthew.a.chan@sony.com"