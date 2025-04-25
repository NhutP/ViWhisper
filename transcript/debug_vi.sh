#!/bin/sh
#SBATCH -o output.out
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:0
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

echo "Job started on $(hostname)"
source /home/mmlab/miniconda3/etc/profile.d/conda.sh

conda activate S2T
echo "Conda environment activated"

export CUDA_VISIBLE_DEVICES=4
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python whisper_transcript_bug_vivos.py
echo "Job completed"