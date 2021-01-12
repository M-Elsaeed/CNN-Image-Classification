ARG UBUNTU_VERSION=18.04

FROM ubuntu:${UBUNTU_VERSION} as base

RUN apt-get update && apt-get install -y curl

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow
ARG TF_PACKAGE_VERSION=
RUN python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}
RUN python3 -m pip install opencv-python keras

COPY bashrc /etc/bash.bashrc

RUN cd /home/
# RUN mkdir 16p8160
# RUN chmod -R 777 16p8160
COPY code/ /home/code/
# RUN chmod -R 777 16p8160
RUN chmod -R 777 /home/code/

RUN chmod a+rwx /etc/bash.bashrc