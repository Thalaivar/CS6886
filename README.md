# CS6886
---
## Assignment 5
The code in `A5/inference.py` runs the latency calculations for the different models. To run the code on a new VM:
1. Make the two setup scripts executable:
```bash
cd ./CS6886/A5
chmod +x *.sh
```
2. Install CUDA-10.2 and Anaconda on the VM with `bash install_conda_cuda.sh`. Fill in the agreement forms and the options for the CUDA installation.
3. [Add](https://askubuntu.com/questions/885610/nvcc-version-command-says-nvcc-is-not-installed) the CUDA installation to PATH
4. Create an Anaconda environment with Python 3.7:
```bash
source ~/.bashrc
conda create -n sysdl python=3.7
conda activate sysdl
```
5. Install CuDNN-8.0.5.39 and required python libraries and download required models: `bash install_stuff.sh`
6. Run the main code:
```bash
cd ~/CS6886/A5/
python inference.py
```

The latencies of each model are outputted to the terminal in a readable format.

**Note:** The above code was run on an Ubuntu 16.04 machine
