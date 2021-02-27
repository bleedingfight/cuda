FROM ubuntu:20.04

USER root:root

ENV LANG C.UTF-8

RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse">/etc/apt/sources.list
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse">>/etc/apt/sources.list
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse">>/etc/apt/sources.list
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse">>/etc/apt/sources.list
RUN rm /etc/apt/sources.list && \
    apt-get update 
RUN apt install  nmap git wget zsh -y
#         libxml2 \
#         git \
#         python3.6-dev \
#         unzip \
#         make \
#         cmake \
#         tree \
#         nmap \
#         libssl-dev && \
#     rm -rf /var/lib/apt/lists/*
#RUN CUDA_TOOLKIT=cuda_10.1.243_418.87.00_linux.run && \
#        wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/$CUDA_TOOLKIT && \
#        sh $CUDA_TOOLKIT --silent --toolkit && \
#        rm $CUDA_TOOLKIT

RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    flask-cors==3.0.9 \
    websockets==8.1
RUN mkdir /usr/local/TensorRT-7.2.1.6
COPY /usr/local/TensorRT-7.2.1.6/python/tensorrt-7.2.1.6-cp37-none-linux_x86_64.whl /opt/tensorRT/python/
