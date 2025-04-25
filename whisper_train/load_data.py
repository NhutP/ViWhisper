from datasets import load_dataset, load_from_disk, concatenate_datasets, interleave_datasets
import psutil
import pathlib
from utils.prepare_data import list_leaf_dirs
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from utils.memory import avoid_OOM
import random
from tqdm import tqdm
import time

# from ..augmentation.augmenter import audio_augmenter

random.seed(1506)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# @dataclass
# class DataCollatorWithOntheflyAugmentation:
#     processor: Any
#     augmentater: audio_augmenter
#     def __call__(self, batch: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         # split inputs and labels since they have to be of different lengths and need different padding methods
#         # first treat the audio inputs by simply returning torch tensors
#         input_features = [{"input_features": self.processor(self.augmentater.augment_ramdom_single(sample["audio"]["array"]), sampling_rate=16000, return_tensors='np').input_features[0]} for sample in batch]
#         batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

#         # get the tokenized label sequences
#         label_features = [{"input_ids": self.processor.tokenizer(sample["transcription"].lower()).input_ids} for sample in batch]
#         # pad the labels to max length
#         labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

#         # replace padding with -100 to ignore loss correctly
#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

#         # if bos token is appended in previous tokenization step,
#         # cut bos token here as it's append later anyways
#         if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
#             labels = labels[:, 1:]

#         batch["labels"] = labels

#         return batch


class whisper_data_loader:
  def __init__(self, data_folder):
    self.data_folder = data_folder
    

  def load_dataset_recur(self, path: pathlib.Path, keep_dataset_in_memory = False):
    leaf_dirs = list_leaf_dirs(path)

    if len(leaf_dirs) > 0:
      concat_dataset = None
      datasets = []
      print(f'Loading dataset at {str(path)}')
      for j in tqdm(leaf_dirs):
        avoid_OOM()
        datasets.append(load_from_disk(str(j), keep_in_memory = keep_dataset_in_memory))
        
      concat_dataset = concatenate_datasets(datasets)
      return concat_dataset
    
    else:
      avoid_OOM()
      dataset = load_from_disk(str(path), keep_in_memory = keep_dataset_in_memory)
      return dataset


  def load_all(self, keep_dataset_in_memory = False):
    # load the all parts of the dataset
    return self.load_dataset_recur(self.data_folder, keep_dataset_in_memory)
  

  def load_remote_train(self, remote_strogage, data_id = ['cmv', 'vivos'], keep_dataset_in_memory = False, is_merge = True):
    print('Load remote train set')
    datasets = [self.load_dataset_recur(str(remote_strogage / i / 'train'), keep_dataset_in_memory) for i in data_id]
    if is_merge:
      return concatenate_datasets(datasets)
    else:
      return datasets
    
  
  def load_remote_eval(self, remote_strogage, data_id = ['cmv', 'vivos'], keep_dataset_in_memory = False):
    print('Load remote eval set')
    eval_datasets = {}
    for i in data_id:
      eval_datasets[i] = self.load_dataset_recur(str(remote_strogage / i / 'validation'), keep_dataset_in_memory)
      if len(eval_datasets[i]) > 3000:
        eval_datasets[i] = eval_datasets[i].select(list(random.sample(range(0, len(eval_datasets[i])), 3000)))
    return eval_datasets