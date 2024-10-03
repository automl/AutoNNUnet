#!/bin/bash


#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120GB
#SBATCH -J evaluate_msd
#SBATCH -t 3-00:00:00                                   
#SBATCH --mail-type fail
#SBATCH --mail-user becktepe@stud.uni-hannover.de      
#SBATCH -p kisski-inference                                         
#SBATCH --output evaluate_msd_%A.out
#SBATCH --error evaluate_msd_%A.err

module load Miniconda3
module load CUDA

conda activate /mnt/home/jbecktep/.conda/envs/automis
export nnUNet_n_proc_DA=30

cd ..
python runscripts/evaluate_msd.py --approach=baseline

