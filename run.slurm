#!/bin/bash
#SBATCH --account=punim0619
#SBATCH --mail-user=wsha@student.unimelb.edu.au
#SBATCH --job-name=unet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=shortgpgpu

module load Keras/2.1.5-intel-2017.u2-GCC-6.2.0-CUDA9-Python-3.6.4-GPU
python main.py
