import sys
sys.path.insert(0, r'..')

import pathlib
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, DatasetDict
from utils.prepare_data import list_leaf_dirs, format_string, list_leaf_dirs_with_extension
from utils.memory import avoid_OOM, clear_audio_folder_cache
import argparse
import pandas as pd


def load_dataset_recur(path: pathlib.Path, cache_dir, keep_dataset_in_memory = False):
  leaf_dirs = list_leaf_dirs_with_extension(path, 'wav')

  if len(leaf_dirs) > 0:
    concat_dataset = None
    datasets = []
    print(f'Loading dataset at {str(path)}')
    for j in tqdm(leaf_dirs):
      # avoid_OOM()    
      datasets.append(load_dataset("audiofolder", data_dir=str(j), keep_in_memory = keep_dataset_in_memory, cache_dir=cache_dir)['train'])
      clear_audio_folder_cache(cache_dir)
    # concat_dataset = concatenate_datasets(datasets)
    # return concat_dataset
    datasets = [i.remove_columns(['is_changed']) if ("is_changed" in i.column_names) else i for i in datasets]
    return datasets
  
  else:
    # avoid_OOM()
    dataset = load_dataset("audiofolder", data_dir=str(path), keep_in_memory = keep_dataset_in_memory, cache_dir=cache_dir)
    return dataset


def prepare_map_for_push(batch):
  batch['audio']['path'] = None
  batch['transcription'] = format_string(batch["transcription"])
  return batch


# if __name__ == '__main__':
#   dataset_list = load_dataset_recur(pathlib.Path('/mnt/mmlab2024/datasets/Backup/final_dataset/wav_split_2/data_1'), cache_dir='/mnt/mmlab2024/datasets/VNSTT/temp_cache/push_cache')

#   mapped_dataset_list = dataset_list
#   print('mapping')
#   # for i in tqdm(range(len(dataset_list))):
#   #   # temp_dataset = dataset_list[i].map(prepare_map_for_push, num_proc=1, writer_batch_size=1000, cache_file_name= '/mnt/mmlab2024/datasets/VNSTT/temp_cache/cache5/temp.arrow')
#   #   # dataset_list[i] = temp_dataset
#   #   mapped_dataset_list.append(dataset_list[i].map(prepare_map_for_push, num_proc=1, writer_batch_size=1000, cache_file_name= '/mnt/mmlab2024/datasets/VNSTT/temp_cache/cache5/temp.arrow'))


#   final_dataset_to_push = concatenate_datasets(mapped_dataset_list).shuffle(1061)
#   # final_dataset_to_push = final_dataset_to_push.map(prepare_map_for_push, num_proc=1, writer_batch_size=1000, cache_file_name= '/mnt/mmlab2024/datasets/VNSTT/temp_cache/cache5/temp.arrow')
#   print(final_dataset_to_push)

#   final_dataset_to_push.push_to_hub("NhutP/VSV-1100_v1", split="train_clean", max_shard_size = "5GB")


if __name__ == '__main__':
  train_clean_list = load_dataset_recur(pathlib.Path('/mnt/mmlab2024/datasets/Backup/final_dataset/wav_split_2'), cache_dir='/mnt/mmlab2024/datasets/VNSTT/temp_cache/push_cache_2')

  train_noise_dataset_list = load_dataset_recur(pathlib.Path('/mnt/mmlab2024/datasets/Backup/final_dataset/splited_ytb_with_perfect'), cache_dir='/mnt/mmlab2024/datasets/VNSTT/temp_cache/push_cache_3')

  train_clean_push_dataset = concatenate_datasets(train_clean_list).shuffle(1061)
  train_noise_push_dataset = concatenate_datasets(train_noise_dataset_list).shuffle(1061)

  

  ddict = DatasetDict({
    "train_clean": train_clean_push_dataset, 
    "train_noise": train_noise_push_dataset
  })

  print(ddict)

  ddict.push_to_hub("NhutP/VSV_1100_v1", max_shard_size = "5GB")
  # final_dataset_to_push.push_to_hub("NhutP/VSV-1100_v1")