
First install docker-ce and nvidia-docker2:
https://docs.docker.com/install/linux/docker-ce/ubuntu/
https://github.com/NVIDIA/nvidia-docker

build the image with:
docker build . -t keras_test

run with
docker run --runtime=nvidia --rm -it -v $PWD/..:/app keras_test
