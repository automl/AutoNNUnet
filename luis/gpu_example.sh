#!/bin/bash

#SBATCH --job-name=test_gpu
#SBATCH --partition=ai,ainlp,tnt
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-cpu=10M
#SBATCH --time=00:5:00
#SBATCH --mail-user=<y.name>@ai.uni-hannover.de
#SBATCH --mail-type=FAIL
#SBATCH --output test_gpu-job_%j.out
#SBATCH --error test_gpu-job_%j.err

module load GCC/10.3.0
module load CMake/3.20.1

conda activate myenv

{%- if cookiecutter.command_line_interface|lower == 'hydra' %}
python cli.py id=$SLURM_ARRAY_TASK_ID{%- else %}
python cli.py --id $SLURM_ARRAY_TASK_ID{% endif %}