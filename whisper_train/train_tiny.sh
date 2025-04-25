#!/bin/sh

export TRANSFORMERS_CACHE=/mnt/mmlab2024/datasets/VNSTT/training_cache
export HF_DATASETS_CACHE=/mnt/mmlab2024/datasets/VNSTT/training_cache
export HF_HOME=/mnt/mmlab2024/datasets/VNSTT/training_cache


source /opt/miniconda3/etc/profile.d/conda.sh
conda activate VNS2T

python train.py tiny /mnt/mmlab2024/datasets/eval/new_data/tiny/20240528191606/checkpoint-15816 /mnt/mmlab2024/datasets/final_data_proccess /mnt/mmlab2024/datasets/final_checkpoint/tiny --rmstrda /mnt/mmlab2024/datasets/remote_data/medium --slda n --nte 3 --evt 0.05 --sat 0.05 --ibs 128 --evibs 128 --evdts cmv14vivos+vlsp+bud500 --dtsbm cmv14vivos


