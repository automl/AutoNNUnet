#!/bin/bash

#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=120GB
#SBATCH -J inference
#SBATCH -t 3-00:00:00                                    
#SBATCH -p kisski-inference                                         
#SBATCH --output runscripts/inference_%A.out
#SBATCH --error runscripts/inference_%A.err

module load Miniforge3
module load CUDA

export nnUNet_n_proc_DA=30

/mnt/home/jbecktep/.conda/envs/automis/bin/python runscripts/run_inference.py --approach=$1

