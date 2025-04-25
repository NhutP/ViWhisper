import sys
sys.path.insert(0, r'..')

import pathlib
import pandas as pd
import soundfile as sf
import numpy as np
import argparse
from datasets import load_dataset, concatenate_datasets

from utils.prepare_data import list_leaf_dirs
from utils.memory import clear_audio_folder_cache
from tqdm import tqdm



class local_combiner:
  def combine(self, train_dataset, save_wav = None, types=['sssss', 'ssss-', 'sss-s', 'sss--', 'ss-ss', 'ss-s-', 'ss--s', 'ss---', 's-sss', 's-ss-', 's-s-s', 's-s--', 's--ss', 's--s-', 's---s', 's----', '-ssss', '-sss-', '-ss-s', '-ss--', '-s-ss', '-s-s-', '-s--s', '-s---', '--sss', '--ss-', '--s-s', '--s--', '---ss'], audio_length=30, sampling_rate=16000):
    '''
    Have bug at the stop condition in the loop
    '''

    num_of_samples = len(train_dataset)

    if save_wav is not None:
      save_wav = pathlib.Path(save_wav)
      (save_wav / 'audio').mkdir(parents=True)
      (save_wav / 'transcript').mkdir(parents=True)

    report = {'file_name': [], 'transcription' : [], 'type' : []}

    audio_array_length = audio_length * sampling_rate
    file_name_index = 1

    audio_current_index = 1

    type_index = 0

    accumulate_sample = False
    
    datset_iterator = iter(train_dataset)

    print('Start combine')

    while audio_current_index < num_of_samples + 1:
      completion = ''
      array_current_index = 0
      type = types[type_index % len(types)]

      combined_audio = np.zeros(audio_array_length, dtype=float)
      combined_transcription = [] 

      for i in type:
        if i == 's':
          if audio_current_index >= num_of_samples + 1:
            break

          if not accumulate_sample:
            audio_current_index += 1
            data = next(datset_iterator)
            # print(audio_current_index)

          data_audio = data['audio']['array']
          next_array_index = array_current_index + data_audio.shape[0]

          if next_array_index <= audio_array_length:
            combined_audio[array_current_index : next_array_index] = data_audio
            combined_transcription.append(data['transcription'])

            array_current_index = next_array_index
            completion += 's'
            accumulate_sample = False
          else:
            accumulate_sample = True
            break

        elif i == '-':
          remain_time = audio_array_length - array_current_index
          array_current_index += np.random.randint(min(remain_time - 1, 5*sampling_rate), min(remain_time, 10*sampling_rate))
          completion += '-'

      type_index += 1
      

      if save_wav is not None and 's' in completion:
        sf.write(str(save_wav / 'audio'/(str(file_name_index) + '.wav')), combined_audio, 16000)
        with open(str(save_wav / 'transcript'/(str(file_name_index) + '.txt')), 'w', encoding='utf8') as f:
          f.write(' '.join(combined_transcription))
        report['file_name'].append(str(file_name_index) + '.wav')
        report['transcription'].append(' '.join(combined_transcription))

        if len(completion) < 5:
          completion += '-' * (5 - len(completion))

        report['type'].append(completion)
        file_name_index += 1

    print('finished')
    df = pd.DataFrame.from_dict(report)
    df.to_csv(str(save_wav / 'audio' / 'metadata.csv'), index=False)
  

  def combine_notrans(self, train_dataset, save_wav = None, types=['sssss', 'ssss-', 'sss-s', 'ss-ss', 's-sss', '-ssss'], audio_length=30, sampling_rate=16000):
    '''
    Have bug at the stop condition in the loop
    '''

    num_of_samples = len(train_dataset)

    if save_wav is not None:
      save_wav = pathlib.Path(save_wav)
      (save_wav / 'audio').mkdir(parents=True)

    report = {'file_name': [], 'type' : []}

    audio_array_length = audio_length * sampling_rate
    file_name_index = 1

    audio_current_index = 1

    type_index = 0

    accumulate_sample = False
    
    datset_iterator = iter(train_dataset)

    print('Start combine')

    while audio_current_index < num_of_samples + 1:
      completion = ''
      array_current_index = 0
      type = types[type_index % len(types)]

      combined_audio = np.zeros(audio_array_length, dtype=float)

      for i in type:
        if i == 's':
          if audio_current_index >= num_of_samples + 1:
            break

          if not accumulate_sample:
            audio_current_index += 1
            data = next(datset_iterator)
            # print(audio_current_index)

          data_audio = data['audio']['array']
          next_array_index = array_current_index + data_audio.shape[0]

          if next_array_index <= audio_array_length:
            combined_audio[array_current_index : next_array_index] = data_audio

            array_current_index = next_array_index
            completion += 's'
            accumulate_sample = False
          else:
            accumulate_sample = True
            break

        elif i == '-':
          remain_time = audio_array_length - array_current_index
          array_current_index += np.random.randint(min(remain_time - 1, 5*sampling_rate), min(remain_time, 7*sampling_rate))
          completion += '-'

      type_index += 1
      

      if save_wav is not None and 's' in completion:
        sf.write(str(save_wav / 'audio'/(str(file_name_index) + '.wav')), combined_audio, 16000)
        report['file_name'].append(str(file_name_index) + '.wav')

        if len(completion) < 5:
          completion += '-' * (5 - len(completion))
        report['type'].append(completion)
        file_name_index += 1

    print('finished')
    # df = pd.DataFrame.from_dict(report)
    # df.to_csv(str(save_wav / 'audio' / 'metadata.csv'), index=False)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data', type=str, help='The path to the input file or directory')
  parser.add_argument('repa', type=str, help='The path to the result file or directory')
  parser.add_argument('--chulen', type=int, help='chunk length', default=30)
  parser.add_argument('--sr', type=int, help='sampling rate', default=16000)
  parser.add_argument('--cap', help='audio folder cache path', type=str)
  parser.add_argument('--label', help = 'process with label or not', default='yes')
  parser.add_argument('--seed', help = 'seed', type = int, default=1061)
  args = parser.parse_args()

  inpa = pathlib.Path(args.data)
  repa = pathlib.Path(args.repa)
  chulen = int(args.chulen)
  with_label = bool(args.label == 'yes')
  seed = args.seed
  # dataset_dirs = list_leaf_dirs(inpa)

  dataset_all = None

  if with_label:
    dataset_dirs = list_leaf_dirs(inpa)
    for i in tqdm(range(len(dataset_dirs))):
      if with_label:
        dataset = load_dataset("audiofolder", data_dir=dataset_dirs[i])['train']
      else:
        dataset = load_dataset("audiofolder", data_dir=dataset_dirs[i], drop_metadata=True, drop_labels=True)['train']
        
      if i != 0:
        dataset_all = concatenate_datasets([dataset_all, dataset])
      else:
        dataset_all = dataset
      
    if not with_label:
      dataset_all = dataset_all.shuffle(seed=seed)
  else:
    dataset_all = load_dataset("audiofolder", data_dir=inpa, drop_labels=True)
    print(dataset_all)
    dataset_all = concatenate_datasets([dataset_all['train'], dataset_all['validation'], dataset_all['test']])
    dataset_all = dataset_all.shuffle(seed=seed)
  # if with_label:
  #   dataset_dirs = list_leaf_dirs(inpa)
  #   for i in tqdm(range(len(dataset_dirs))):
  #     # print(dataset_dirs[i])
  #     if with_label:
  #       dataset = load_dataset("audiofolder", data_dir=dataset_dirs[i])['train']
  #     else:
  #       dataset = load_dataset("audiofolder", data_dir=dataset_dirs[i], drop_metadata=True, drop_labels=True)['train']
        
  #     if i != 0:
  #       dataset_all = concatenate_datasets([dataset_all, dataset])
  #     else:
  #       dataset_all = dataset
      
  #   if not with_label:
  #     dataset_all = dataset_all.shuffle(seed=22521061)


  print("Info:")
  print(dataset_all)

  combiner = local_combiner()
  if with_label:
    print("Combine with transcription")
    combiner.combine(dataset_all, repa)
  else:
    print("Combine without transcription")
    combiner.combine_notrans(dataset_all, repa, audio_length=chulen)