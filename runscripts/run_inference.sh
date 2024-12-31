#!/bin/bash


#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=120GB
#SBATCH -J inference
#SBATCH -t 3-00:00:00                                   
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de      
#SBATCH -p kisski-inference                                         
#SBATCH --output inference_%A.out
#SBATCH --error inference_%A.err

module load Miniforge3
module load CUDA

export nnUNet_n_proc_DA=30

cd ..
/mnt/home/jbecktep/.conda/envs/automis/bin/python runscripts/run_inference.py --approach=baseline_ResidualEncoderL

