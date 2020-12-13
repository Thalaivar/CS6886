#!/bin/bash

sudo apt update -y && sudo apt upgrade -y
sudo apt install gcc make g++ -y

cd ~/
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run -O cuda_install.run
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O anaconda_install.sh

bash anaconda_install.sh
sudo sh cuda_install.run