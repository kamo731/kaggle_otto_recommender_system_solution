#!/bin/bash
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo apt update -y
sudo apt upgrade -y
sudo python3 install_gpu_driver.py