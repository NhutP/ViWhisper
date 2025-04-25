from local_format import local_dataset_processor
import numpy as np
from datasets import Dataset, Features, Audio, load_dataset
import soundfile as sf 
import pathlib
import argparse
import pandas as pd
from transformers import WhisperProcessor
import shutil
import pandas as pd


global cache_path


class remote_data_process_core(local_dataset_processor):
  def __init__(self, processor):
    super().__init__(processor)


  def process_save_remote_dataset(self, dataset_dict, save_path):
    for name, data_split in dataset_dict.items():
      data_split = data_split.cast_column("audio", Audio(sampling_rate=16000))
      data_split = data_split.map(self.prepare_dataset, batch_size=500, remove_columns=data_split.column_names)
      data_split.save_to_disk(str(save_path / name), num_proc=10)

  
  def combine(self, train_dataset, num_of_samples, save_mp3 = None, start_index = 0, types=['s---', '-s--', '--s-', '---s', 's-s-', '-s-s', 'ss--', '-ss-', '--ss', 'sss-', '-sss'], audio_length=30, sampling_rate=16000, sample_per_save = 5000):
    '''
    Have bug at the stop condition in the loop
    '''
    if save_mp3 is not None:
      save_mp3 = pathlib.Path(save_mp3)
      (save_mp3 / 'audio').mkdir(parents=True)
      (save_mp3 / 'transcript').mkdir(parents=True)

    report = {'file_name': [], 'transcription' : [], 'type' : []}

    audio_array_length = audio_length * sampling_rate
    file_name_index = 1
    
    audio_current_index = 1
    created_files = [int(i.name) for i in list((save_mp3 / 'audio').rglob('*.mp3'))]
    audio_current_index = max(created_files) if len(created_files) > 0 else 1

    type_index = 0

    accumulate_sample = False
    
    datset_iterator = iter(train_dataset)

    if start_index > 0:
      print(f'Skip index, start at {start_index}')

    for i in range(start_index):
      _ = next(datset_iterator)

    print('Start combine')

    while audio_current_index < num_of_samples + 1:
      completion = ''
      array_current_index = 0
      type = types[type_index % len(types)]

      combined_audio = np.zeros(audio_array_length, dtype=float)
      combined_transcription = [] 

      for i in type:
        if i == 's':
          if audio_current_index >= num_of_samples:
            break

          if not accumulate_sample:
            audio_current_index += 1
            data = next(datset_iterator)
            print(audio_current_index)

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
      

      if save_mp3 is not None:
        sf.write(str(save_mp3 / 'audio'/(str(file_name_index) + '.mp3')), combined_audio, 16000)
        with open(str(save_mp3 / 'transcript'/(str(file_name_index) + '.txt')), 'w', encoding='utf8') as f:
          f.write(' '.join(combined_transcription))
        report['file_name'].append(str(file_name_index) + '.mp3')
        report['transcription'].append(' '.join(combined_transcription))
        report['type'].append(completion)
        file_name_index += 1

    df = pd.DataFrame.from_dict(report)
    df.to_csv(str(save_mp3 / 'audio' / 'metadata.csv'), index=False)


  def save_mp3_streaming_mode(self, dataset, save_path):
    meta = {'file_name' : [], 'transcription' : []}
    file_name_index = 1

    (save_path / 'audio').mkdir(parents=True)
    (save_path / 'transcript').mkdir(parents=True)

    for sample in dataset:
      sf.write(str(save_path / 'audio'/(str(file_name_index) + '.mp3')), sample['audio']['array'], 16000)
      with open(str(save_path / 'transcript'/(str(file_name_index) + '.txt')), 'w', encoding='utf8') as f:
        f.write(sample['transcription'])

      meta['file_name'].append((str(file_name_index) + '.mp3'))
      meta['transcription'].append(sample['transcription'])

      file_name_index += 1
    
    df = pd.DataFrame.from_dict(meta)
    df.to_csv(str(save_path / 'audio' / 'metadata.csv'), index=False)



class remote_data_processor(remote_data_process_core):
  def __init__(self, processor):
    super().__init__(processor)

  
  def load_process_remote_dataset(self, save_folder, id = ['cmv', 'vivos']):
    if 'cmv' in id:
      if not (save_folder / 'cmv').exists():
        split = ['train', 'validation', 'test']
        print('Processing CMV')
        cmv_dataset = {}
        for i in split:
          cmv_dataset[i] = load_dataset("mozilla-foundation/common_voice_14_0", "vi", split=i, use_auth_token=True).rename_column("sentence", "transcription")

        print(f'Info: {cmv_dataset}')
        
        self.process_save_remote_dataset(cmv_dataset, save_folder / 'cmv')

      else:
        print('Already loaded and processed CMV')


    if 'vivos' in id:
      if not (save_folder / 'vivos').exists():
        split = ['train', 'test']
        print('Processing vivos')
        vivos_dataset = {}

        for i in split:
          dataset = load_dataset("vivos", split=i, use_auth_token=True).rename_column("sentence", "transcription")
          if i == "train":
            dataset = dataset.train_test_split(test_size=0.03)
            vivos_dataset[i] = dataset['train']
            vivos_dataset['validation'] = dataset['test']
          else:
            vivos_dataset[i] = dataset

        print(f"Info: {vivos_dataset}")

        self.process_save_remote_dataset(vivos_dataset, save_folder / 'vivos')
      else:
        print("Already loaded and processed vivos")



  def load_combine_process_remote_dataset(self, save_folder, start_index=0, id = ['bud500']):
    if "bud500" in id:
      if not (save_folder / "Bud500").exists():
        print('Processing bud500')
        bud_dataset = {}

        split = ['test', 'validation', 'train']
        num_of_samples = [7500, 7500, 634158]

        for i in range(len(split)):
          # validation and test
          if split[i] == 'validation' or split[i] == 'test':
            if (save_folder / "Bud500" / split[i]).exists():
              continue
            print(f"Bud 500 {split[i]} set")
            bud_dataset = load_dataset("linhtran92/viet_bud500", split=split[i], use_auth_token=True, streaming=True)
            self.save_mp3_streaming_mode(bud_dataset, save_folder / "Bud500" / split[i])
  

          elif split[i] == 'train':
            print(f"Bud 500 {split[i]} set")
            bud_dataset = load_dataset("linhtran92/viet_bud500", split=split[i], use_auth_token=True, streaming=True)
            self.combine(bud_dataset, num_of_samples[i], save_folder / "Bud500" / split[i], start_index)

        print(f"Info: {bud_dataset}")
      else:
        print('Already loaded and processed bud500')



if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('pv', help = 'processor version (tiny, base, small, medium, large)', choices=['tiny', 'base', 'small', 'medium', 'large'])
  parser.add_argument('stro', help='path to processed data strogage folder it has 3 sub folders: train, validation, test', type=str)
  parser.add_argument('--task', help= '0 if process original, 1 if combine', type=str, default='')
  parser.add_argument('--rmdtna', help="remote data name (ex: 'cmv+vicos+bud500']) for original processing or combine processing for test and validaiton", default= 'cmv+vicos+bud500')
  parser.add_argument('--stin', help='Start index for large dataset like BUD500', default=0, type=int)

  args = parser.parse_args()

  processor_version = args.pv
  strogage_folder = pathlib.Path(args.stro)
  remote_datasets = str(args.rmdtna).split('+')
  task = args.task
  start_index = int(args.stin)
  
  model_id =  "openai/whisper-" + processor_version
  processor = WhisperProcessor.from_pretrained(model_id, language="vi", task="transcribe")
  
  remote_data_format = remote_data_processor(processor)

  if task == '0':
    remote_data_format.load_process_remote_dataset(strogage_folder, remote_datasets)
  elif task == '1':
    remote_data_format.load_combine_process_remote_dataset(strogage_folder, start_index)