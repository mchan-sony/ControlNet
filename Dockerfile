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
    && conda env create -f environment.yaml \
    && python -m pip install --upgrade pip  \
    && pip install -y gradio \ 
    && pip3 install -U -y torchvision --index-url https://download.pytorch.org/whl/cu118

RUN git config --global user.name "Matthew Chan" \
    && git config --global user.email "matthew.a.chan@sony.com" \
    && git lfs install

# download model checkpoints
RUN cd /workspace/controlnet \
    && git clone https://huggingface.co/lllyasviel/ControlNet weights \
    && mv weights/annotator/ckpts/* annotator/ckpts \
    && mv weights/models/* models \
    && mv weights/training . \
    && rm -rf weights
