#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=RMU_eval
#SBATCH --requeue
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=mhong


# # Load required modules
# module load gcc/11.3.0
# module load conda
module load cuda/12.1.1
# module list

# # Activate conda environment
# source activate muse_bench

# Benchmark info
echo "GPU availability:"
nvidia-smi
echo ""
echo "Python executable:"
which python3
echo ""
echo "TIMING - Starting unlearning at: $(date)"
echo "Job is starting on $(hostname)"
echo ""

# Set up HuggingFace (update with your token and path) #SBATCH --gres=gpu:a100:1
# export 'HF_TOKEN=your_hf_token'
# export HF_HOME="your_hf_home_path"
# huggingface-cli login --token $HF_TOKEN



CUDA_VISIBLE_DEVICES=1 python whole_eval.py 
