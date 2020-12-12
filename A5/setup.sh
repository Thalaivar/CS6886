#!/bin/bash

sudo apt update -y && sudo apt upgrade -y

wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin 

sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /"
sudo apt-get update

sudo apt-get install libcudnn8=8.0.5.39-1+cuda10.2
sudo apt-get install libcudnn8-dev=8.0.5.39-1+cuda10.2

wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

mkdir ~/models && cd ~/models
wget https://zenodo.org/record/2592612/files/resnet50_v1.onnx
wget https://zenodo.org/record/3157894/files/mobilenet_v1_1.0_224.onnx
wget https://zenodo.org/record/3163026/files/ssd_mobilenet_v1_coco_2018_01_28.onnx
wget https://zenodo.org/record/3228411/files/resnet34-ssd1200.onnx
wget https://zenodo.org/record/2581623/files/model_best.pth?download=1 -O gnmt_wts.pth