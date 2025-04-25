#!/bin/bash

# # Command to be executed
# COMMAND="python augmenter.py /mnt/mmlab2024/datasets/train_dataset /mnt/mmlab2024/datasets/train_dataset_aug --bgstr /mnt/mmlab2024/datasets/VNSTT/back_ground_final/use_bgn --csvp /mnt/mmlab2024/datasets/final_augment.csv --geninfo no"

COMMAND1Y="python augmenter.py /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_raw /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_augment/augment_2252 --bgstr /mnt/mmlab2024/datasets/VNSTT/back_ground_final/use_bgn --csvp /mnt/mmlab2024/datasets/ViWhisper/assets/vietvivos_fixed_augment_infor/seed_2252.csv --geninfo yes --augpnf 2 --seed 2252"

COMMAND2Y="python augmenter.py /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_raw /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_augment/augment_1061 --bgstr /mnt/mmlab2024/datasets/VNSTT/back_ground_final/use_bgn --csvp /mnt/mmlab2024/datasets/ViWhisper/assets/vietvivos_fixed_augment_infor/seed_1061.csv --geninfo yes --augpnf 2 --seed 1061"

COMMAND3Y="python augmenter.py /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_raw /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_augment/augment_1506 --bgstr /mnt/mmlab2024/datasets/VNSTT/back_ground_final/use_bgn --csvp /mnt/mmlab2024/datasets/ViWhisper/assets/vietvivos_fixed_augment_infor/seed_1506.csv --geninfo yes --augpnf 2 --seed 1506"

COMMAND4Y="python augmenter.py /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_raw /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_augment/augment_1803 --bgstr /mnt/mmlab2024/datasets/VNSTT/back_ground_final/use_bgn --csvp /mnt/mmlab2024/datasets/ViWhisper/assets/vietvivos_fixed_augment_infor/seed_1803.csv --geninfo yes --augpnf 2 --seed 1803"

COMMAND5Y="python augmenter.py /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_raw /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_augment/augment_2004 --bgstr /mnt/mmlab2024/datasets/VNSTT/back_ground_final/use_bgn --csvp /mnt/mmlab2024/datasets/ViWhisper/assets/vietvivos_fixed_augment_infor/seed_2004.csv --geninfo yes --augpnf 2 --seed 2004"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/mmlab2024/datasets/conda/VNS2T

# # Loop until the command succeeds
# while true; do
#   # Execute the command
#   $COMMAND1
#   $COMMAND2
#   $COMMAND3
#   $COMMAND4
#   $COMMAND5
# done


$COMMAND1Y
$COMMAND2Y
$COMMAND3Y
$COMMAND4Y
$COMMAND5Y


COMMAND1N="python augmenter.py /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_raw /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_augment/augment_2252 --bgstr /mnt/mmlab2024/datasets/VNSTT/back_ground_final/use_bgn --csvp /mnt/mmlab2024/datasets/ViWhisper/assets/vietvivos_fixed_augment_infor/seed_2252.csv --geninfo no --augpnf 2 --seed 2252"

COMMAND2N="python augmenter.py /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_raw /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_augment/augment_1061 --bgstr /mnt/mmlab2024/datasets/VNSTT/back_ground_final/use_bgn --csvp /mnt/mmlab2024/datasets/ViWhisper/assets/vietvivos_fixed_augment_infor/seed_1061.csv --geninfo no --augpnf 2 --seed 1061"

COMMAND3N="python augmenter.py /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_raw /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_augment/augment_1506 --bgstr /mnt/mmlab2024/datasets/VNSTT/back_ground_final/use_bgn --csvp /mnt/mmlab2024/datasets/ViWhisper/assets/vietvivos_fixed_augment_infor/seed_1506.csv --geninfo no --augpnf 2 --seed 1506"

COMMAND4N="python augmenter.py /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_raw /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_augment/augment_1803 --bgstr /mnt/mmlab2024/datasets/VNSTT/back_ground_final/use_bgn --csvp /mnt/mmlab2024/datasets/ViWhisper/assets/vietvivos_fixed_augment_infor/seed_1803.csv --geninfo no --augpnf 2 --seed 1803"

COMMAND5N="python augmenter.py /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_raw /mnt/mmlab2024/datasets/VNSTT/vietvivos_fixed_augment/augment_2004 --bgstr /mnt/mmlab2024/datasets/VNSTT/back_ground_final/use_bgn --csvp /mnt/mmlab2024/datasets/ViWhisper/assets/vietvivos_fixed_augment_infor/seed_2004.csv --geninfo no --augpnf 2 --seed 2004"

# Loop until the command succeeds
while true; do
  # Execute the command
  $COMMAND1N
  $COMMAND2N
  $COMMAND3N
  $COMMAND4N
  $COMMAND5N
done