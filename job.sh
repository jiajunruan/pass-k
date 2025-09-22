#!/bin/bash
#SBATCH -p mhong
#SBATCH --job-name=forget
#SBATCH --output=pass.out
#SBATCH --error=pass.err
#SBATCH --gres=gpu:8
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G

# Load necessary modules if required by your cluster
# For example, to load CUDA:
# module load cuda/11.8

# Activate your Conda environments
# Replace 'my_conda_env' with the actual name of your Conda environment
source ~/.bashrc
conda activate unlearning
# Run your Python script
# Make sure 'know.py' is the correct name of your file
python pass-k.py