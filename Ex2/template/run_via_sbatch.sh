#!/bin/bash
#SBATCH --gres gpu:0
#SBATCH --mem 5G
#SBATCH --cpus-per-task 1
#SBATCH --time 30:00
#SBATCH -p exercise
#SBATCH -o slurm_output.log

echo "Loading conda env"
module load anaconda/3
source ~/.bashrc
conda activate EML_ex01

echo "Running exercise02_template.py"
python exercise02_template.py
