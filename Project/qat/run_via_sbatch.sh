#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 10G
#SBATCH --cpus-per-task 4
#SBATCH --time 00:30:00
#SBATCH -p exercise
#SBATCH -o slurm_output.log

export WANDB_API_KEY=6543fc90dd40bb608e2669b7f5082297acfcee74

echo "Loading conda env"
module load anaconda/3
source ~/.bashrc
conda activate prj

echo "Running main.py"
python main.py
