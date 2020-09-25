# Docker mnist show

## Install docker and nvidia-docker2

[Follow the steps here](https://cnvrg.io/how-to-setup-docker-and-nvidia-docker-2-0-on-ubuntu-18-04/)

## Docker Tutorial

[Orientation and setup](https://docs.docker.com/get-started/)


## Running gpu docker image

bash runDocker_gpu.sh
 
## Running cpu docker image

bash runDocker_cpu.sh
 
 
## Running imshow example 

python3 show.py

## Running MNIST training example

python train.py



## Building Docker Image and pushing to Dockerhub

### GPU Image

```
docker build -t jerinign/opencv_tf:gpu . -f Dockerfile.gpu

docker push jerinign/opencv_tf:gpu
```
### CPU Image
```
docker build -t jerinign/opencv_tf:cpu . -f Dockerfile.cpu

docker push jerinign/opencv_tf:cpu
```

