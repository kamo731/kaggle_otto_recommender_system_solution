# FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04
FROM rapidsai/rapidsai-core:22.12-cuda11.5-base-ubuntu20.04-py3.8

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get update -y
RUN apt-get -y clean all
RUN apt-get -y install \
    python3.8 \
    python3-pip \
    vim \
    wget \
    curl \
    git \
    unzip \
    sudo \
    zsh \
    gcc \
    g++ \
    make \
    libsm6 libxrender1
RUN rm -rf /var/lib/apt/lists/*

# cudfのinstall。nvidia/cudaのimageをベースにする場合必要。
# RUN pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.ngc.nvidia.com
# RUN pip install cuml-cu11 --extra-index-url=https://pypi.ngc.nvidia.com
# RUN pip install cugraph-cu11 --extra-index-url=https://pypi.ngc.nvidia.com

COPY requirements.txt .
RUN pip3 install -U pip && pip3 install -r requirements.txt
