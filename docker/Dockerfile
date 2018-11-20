ARG cuda_version=9.0
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

RUN apt-get update && apt-get -y install python3-pip python3-dev

COPY requirements.txt /root
RUN pip3 install -r /root/requirements.txt

WORKDIR /app

