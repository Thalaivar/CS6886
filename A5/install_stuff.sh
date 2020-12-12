#!/bin/bash

# activate the conda environment of your choice before beginning

cd ~/
rm cuda_install.run 
rm anaconda_install.sh

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin 

sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /"
sudo apt-get update

sudo apt-get install libcudnn8=8.0.5.39-1+cuda10.2
sudo apt-get install libcudnn8-dev=8.0.5.39-1+cuda10.2

sudo apt autoremove -y

mkdir ~/models && cd ~/models
wget https://zenodo.org/record/2592612/files/resnet50_v1.onnx
wget https://zenodo.org/record/3157894/files/mobilenet_v1_1.0_224.onnx
wget https://zenodo.org/record/3163026/files/ssd_mobilenet_v1_coco_2018_01_28.onnx
wget https://zenodo.org/record/3228411/files/resnet34-ssd1200.onnx


conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install fastBPE sacremoses subword_nmt omegaconf requests opencv-python onnx onnxruntime-gpu

cd ~/
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./