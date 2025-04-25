#!/bin/sh
#SBATCH -o output_med.out
#SBATCH --time=72:00:00 #time limit to batch job
#SBATCH --gres=gpu:0
#SBATCH -N 1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2


export TRANSFORMERS_CACHE=/data1/mmlab/dataset/temp/training_cache
export HF_DATASETS_CACHE=/data1/mmlab/dataset/temp/training_cache
export HF_HOME=/data1/mmlab/dataset/temp/training_cache


source /home/mmlab/miniconda3/etc/profile.d/conda.sh
conda activate S2T

export CUDA_VISIBLE_DEVICES=3,4,7


torchrun --nproc_per_node=3 --master_port=25678 train.py medium /data1/mmlab/dataset/eval/test_medium/20240528193042/checkpoint-17134 /data1/mmlab/dataset/new_dataset_processed /data1/mmlab/dataset/final_checkpoint/medium --rmstrda /data1/mmlab/dataset/remote_data/medium --slda n --nte 3 --evt 0.05 --sat 0.05 --ibs 128 --evibs 128 --evdts cmv14vivos+vlsp+bud500 --dtsbm cmv14vivos