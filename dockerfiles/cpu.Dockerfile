# 16P8160 Customized Docker File
ARG UBUNTU_VERSION=18.04

FROM ubuntu:${UBUNTU_VERSION} as base

RUN apt-get update && apt-get install -y curl

ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

RUN ln -s $(which python3) /usr/local/bin/python

ARG TF_PACKAGE=tensorflow
ARG TF_PACKAGE_VERSION=
RUN python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}
RUN python3 -m pip install opencv-python keras silence-tensorflow

COPY bashrc /etc/bash.bashrc

RUN cd /home/
COPY code/ /home/
RUN chmod -R 777 /home/

RUN chmod a+rwx /etc/bash.bashrc