#!/bin/sh
#SBATCH -o output_eval.out
#SBATCH --time=72:00:00 #time limit to batch job
#SBATCH --gres=gpu:0
#SBATCH -N 1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2


export TRANSFORMERS_CACHE=/data1/mmlab/dataset/temp/eval_cache
export HF_DATASETS_CACHE=/data1/mmlab/dataset/temp/eval_cache
export HF_HOME=/data1/mmlab/dataset/temp/eval_cache


source /home/mmlab/miniconda3/etc/profile.d/conda.sh
conda activate S2T

export CUDA_VISIBLE_DEVICES=2,3,4,5


torchrun --nproc_per_node=4 --master_port=25671 eval.py small /data1/mmlab/dataset/final_checkpoint/small/20240630150430/checkpoint-42642 /data1/mmlab/dataset/final_eval/small --evdts bud500+cmv14+vivos+vlsp2020_task1+vlsp2020_task2+vlsp2021_task1+vlsp2021_task2 --evdaf /data1/mmlab/dataset/remote_data/medium --evibs 128

# torchrun --nproc_per_node=4 --master_port=25671 eval.py small vinai/PhoWhisper-small /data1/mmlab/dataset/final_eval/temp_ori --evdts bud500+cmv14+vivos+vlsp2020_task1+vlsp2020_task2+vlsp2021_task1+vlsp2021_task2 --evdaf /data1/mmlab/dataset/remote_data/medium --evibs 32
# torchrun --nproc_per_node=4 --master_port=25671 eval.py base vinai/PhoWhisper-base /data1/mmlab/dataset/final_eval/temp_ori --evdts bud500+cmv14+vivos+vlsp2020_task1+vlsp2020_task2+vlsp2021_task1+vlsp2021_task2 --evdaf /data1/mmlab/dataset/remote_data/medium --evibs 64

export CUDA_VISIBLE_DEVICES=2,3,4,5
torchrun --nproc_per_node=4 --master_port=25671 eval.py tiny vinai/PhoWhisper-tiny /data1/mmlab/dataset/final_eval/temp_ori --evdts bud500+cmv14+vivos+vlsp2020_task1+vlsp2020_task2+vlsp2021_task1+vlsp2021_task2 --evdaf /data1/mmlab/dataset/remote_data/medium --evibs 256

export CUDA_VISIBLE_DEVICES=2,3,4,5
torchrun --nproc_per_node=4 --master_port=25671 eval.py medium openai/whisper-medium /data1/mmlab/dataset/final_eval/temp_ori --evdts bud500+cmv14+vivos+vlsp2020_task1+vlsp2020_task2+vlsp2021_task1+vlsp2021_task2 --evdaf /data1/mmlab/dataset/remote_data/medium --evibs 32

export CUDA_VISIBLE_DEVICES=2,3,4,5
torchrun --nproc_per_node=4 --master_port=25671 eval.py small openai/whisper-small /data1/mmlab/dataset/final_eval/temp_ori --evdts bud500+cmv14+vivos+vlsp2020_task1+vlsp2020_task2+vlsp2021_task1+vlsp2021_task2 --evdaf /data1/mmlab/dataset/remote_data/medium --evibs 64

export CUDA_VISIBLE_DEVICES=2,3,4,5
torchrun --nproc_per_node=4 --master_port=25671 eval.py base openai/whisper-base /data1/mmlab/dataset/final_eval/temp_ori --evdts bud500+cmv14+vivos+vlsp2020_task1+vlsp2020_task2+vlsp2021_task1+vlsp2021_task2 --evdaf /data1/mmlab/dataset/remote_data/medium --evibs 128

export CUDA_VISIBLE_DEVICES=2,3,4,5
torchrun --nproc_per_node=4 --master_port=25671 eval.py tiny openai/whisper-tiny /data1/mmlab/dataset/final_eval/temp_ori --evdts bud500+cmv14+vivos+vlsp2020_task1+vlsp2020_task2+vlsp2021_task1+vlsp2021_task2 --evdaf /data1/mmlab/dataset/remote_data/medium --evibs 256

export CUDA_VISIBLE_DEVICES=2,3,4,5
torchrun --nproc_per_node=4 --master_port=25671 eval.py large vinai/PhoWhisper-large /data1/mmlab/dataset/final_eval/temp_ori --evdts bud500+cmv14+vivos+vlsp2020_task1+vlsp2020_task2+vlsp2021_task1+vlsp2021_task2 --evdaf /data1/mmlab/dataset/remote_data/medium --evibs 16