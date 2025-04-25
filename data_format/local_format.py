import sys
sys.path.insert(0, r'..')

import datasets
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import WhisperProcessor

import librosa
import pathlib
import time

import pandas as pd
import pathlib
import shutil
import argparse
import math

from utils.memory import avoid_OOM, clear_audio_folder_cache
from utils.prepare_data import list_leaf_dirs, number_of_subdir_layer, format_string

space = '-------------------------------------------------------'

global cache_path


class local_data_formatter:
  def load_data_single(self, audio_folder, aligned_folder, take_gap=False):
    '''
    exist_audio in return is a list of absolute paths
    the audio folder must be the leaf to consider (it only contains 3 subdir (which are the threshold folder with transcript file))
    '''
    
    transcripted_audio = []
    data_script = {'file_name' : [], 'transcription' : []}
    
    for i in aligned_folder.iterdir():
      nogap_folder = i / 'nogap'
      gap_folder = i / 'gap'

      if (not nogap_folder.exists()) and (not gap_folder.exists()):
        for j in i.iterdir():
          data_script['file_name'].append((j.stem + '.wav'))
          # data_script['transcription'].append(" ".join(j.read_text(encoding='utf8').split()))
          data_script['transcription'].append(format_string(j.read_text(encoding='utf8')))
          transcripted_audio.append(str(audio_folder / (j.stem + '.wav')))

      if nogap_folder.exists():
        for j in nogap_folder.iterdir():
          data_script['file_name'].append((j.stem + '.wav'))
          # data_script['transcription'].append(" ".join(j.read_text(encoding='utf8').split()))
          data_script['transcription'].append(format_string(j.read_text(encoding='utf8')))
          transcripted_audio.append(str(audio_folder / (j.stem + '.wav')))

      if take_gap and gap_folder.exists():
        for j in gap_folder.iterdir():
          data_script['file_name'].append((j.stem + '.wav'))
          # data_script['transcription'].append(" ".join(j.read_text(encoding='utf8').replace('-','').split()))
          data_script['transcription'].append(format_string(j.read_text(encoding='utf8')))
          transcripted_audio.append(str(audio_folder / (j.stem + '.wav')))

    return transcripted_audio, data_script


  def save_meta(self, metadata, save_place):
    metadata = pd.DataFrame.from_dict(metadata)
    metadata.to_csv(pathlib.Path(save_place / 'metadata.csv'), index=False, encoding='utf8')


  def delete_unaligned(self, audio_path, transcripted_audio_list):
    '''
    Note that audio_path is a folder to consider, it only contain the threshold folders, each threshold folder contain gap (and nogap)
    '''
    # delete unaligned files in folder
    available_audio = [str(i) for i in pathlib.Path(audio_path).rglob('*.wav')]
    # print(transcripted_audio_list)
    delete_file  = list(set(available_audio).difference(transcripted_audio_list))

    
    for k in delete_file:
      pathlib.Path(k).unlink()

    print(f"Deleted unaligned files in {audio_path}")


  def format_folder(self, data_folder: pathlib.Path, aligned_folder: pathlib.Path, delete_unaligned=False, take_gap=True):
    corresponding_aligned_folders_from_audio = [pathlib.Path(str(i).replace(str(data_folder), str(aligned_folder))) for i in list_leaf_dirs(data_folder)] 
    available_aligned_folders = list(pathlib.Path(i) for i in aligned_folder.rglob('*') if i.is_dir() and number_of_subdir_layer(i) == 3)

    for i in available_aligned_folders:
      if len(list(i.rglob("*.txt"))) == 0:
        shutil.rmtree(i)
        print(f"Deleted unvalid aligned folder {i}  (folder with 0 txt file)")
        print(space)

    available_aligned_folders = list(pathlib.Path(i) for i in aligned_folder.rglob('*') if i.is_dir() and number_of_subdir_layer(i) == 3)


    delete_audio_folder  = list(set(corresponding_aligned_folders_from_audio).difference(available_aligned_folders))
    delete_audio_folder = list(pathlib.Path(str(i).replace(str(aligned_folder), str(data_folder))) for i in delete_audio_folder)

    for i in delete_audio_folder:
      shutil.rmtree(i)
      print(f"Deleted unaligned audio folder {i}")
      print(space)
    
    corresponding_aligned_folders_from_audio = [pathlib.Path(str(i).replace(str(data_folder), str(aligned_folder))) for i in list_leaf_dirs(data_folder)] 

    audio_folders = [pathlib.Path(str(i).replace(str(aligned_folder), str(data_folder))) for i in available_aligned_folders]
    
    for i in tqdm(range(len(available_aligned_folders))):
      transcripted_audios, data = self.load_data_single(audio_folders[i], available_aligned_folders[i], take_gap)

      if  delete_unaligned:
        self.delete_unaligned(audio_folders[i], transcripted_audios)

      self.save_meta(data, audio_folders[i])



class local_dataset_processor:
  def __init__(self, processor, with_label=True):
    super().__init__()
    self.processor = processor
    self.with_label = with_label

    def prepare_dataset_with_label(batch):
      audio = batch["audio"]
      
      # compute log-Mel input features from input audio array
      batch["input_features"] = self.processor(audio["array"], sampling_rate=16000, return_tensors='np').input_features[0]

      # encode target text to label ids
      # batch["labels"] = self.processor.tokenizer(batch["transcription"].lower()).input_ids
      batch["labels"] = self.processor.tokenizer(format_string(batch["transcription"])).input_ids
      
      return batch
    
    def prepare_map_for_transcription(batch):
      audio = batch["audio"]
      batch['array'] = audio["array"]
      batch['path'] = audio['path']
      return batch

    if with_label:
      self.prepare_dataset = prepare_dataset_with_label
      print("Prepare dataset with label")
    else:
      self.prepare_dataset = prepare_map_for_transcription
      print("Prepare dataset without label")

  def process(self, dataset, batch_size=500, num_proc=10):
    avoid_OOM()
    return dataset.map(self.prepare_dataset, batch_size=batch_size, remove_columns=dataset.column_names["train"], num_proc=num_proc)


  def segment_dataset(self, dataset, num_row):
    print("Segment dataset")
    
    for i in range(0, math.ceil(len(dataset) / num_row)):
      segment = dataset.select(list(range(i * num_row, min((i * num_row + num_row), len(dataset)))))
      yield segment


  def process_save(self, audio_folder, save_folder, cache_file_name_map=None, batch_size=500, num_proc=10, num_writer = 1000, keep_in_memory = False):
    # audio_folder must contain metadata.csv
    if self.with_label:
      dataset = load_dataset("audiofolder", data_dir=audio_folder, keep_in_memory=keep_in_memory, cache_dir = cache_path)
    else:
      dataset =  load_dataset("audiofolder", data_dir=audio_folder, keep_in_memory=keep_in_memory, cache_dir = cache_path, drop_metadata=True, drop_labels=True)

    if len(dataset) == 0:
      print(f"{audio_folder} ia an empty dataset")
      return
    
    dataset = dataset['train']
    print(dataset)

    avoid_OOM()
    if cache_file_name_map is not None:
      dataset = dataset.map(self.prepare_dataset, remove_columns=dataset.column_names, num_proc=num_proc,batch_size=batch_size, cache_file_name = str(cache_file_name_map / 'temp.arrow'), writer_batch_size=num_writer)
    else:
      dataset = dataset.map(self.prepare_dataset, remove_columns=dataset.column_names, num_proc=num_proc,batch_size=batch_size, writer_batch_size=num_writer)

    print("Info:")
    print(dataset)

    avoid_OOM()
    dataset.save_to_disk(save_folder, num_proc=num_proc)
    
    if cache_file_name_map is not None:
      shutil.rmtree(str(cache_file_name_map), ignore_errors=True)

    clear_audio_folder_cache(cache_path)


  def process_save_segment_mode(self, audio_folder, save_folder, cache_file_name_map=None, num_row=2500, batch_size=500, num_proc=10, num_writer=1000, keep_in_memory = False):
    # audio_folder must contain metadata.csv
    if self.with_label:
      dataset = load_dataset("audiofolder", data_dir = audio_folder, keep_in_memory = keep_in_memory, cache_dir = cache_path)
    else:
      dataset =  load_dataset("audiofolder", data_dir=audio_folder, keep_in_memory=keep_in_memory, cache_dir = cache_path, drop_metadata=True, drop_labels=True)

    if len(dataset) == 0:
      print(f"{audio_folder} ia an empty dataset")
      return
    
    dataset = dataset['train']
    print(dataset)

    yield_dataset = self.segment_dataset(dataset, num_row=num_row)
    yield_dataset_2 = self.segment_dataset(dataset, num_row=num_row)

    print('Total rows')
    print(sum([len(i) for i in yield_dataset_2]))
    
    index = 0
    for temp_dataset in yield_dataset:
      if (save_folder / str(index)).exists():
        print(f"{(save_folder / str(index))} exists")
        index += 1
        continue

      (save_folder / str(index)).mkdir()

      print(f"Creating {(save_folder / str(index))}")
      avoid_OOM()
      if cache_file_name_map is not None:
        temp_dataset = temp_dataset.map(self.prepare_dataset, remove_columns=temp_dataset.column_names, num_proc=num_proc,batch_size=batch_size, cache_file_name = str(cache_file_name_map / 'temp.arrow'), writer_batch_size=num_writer)
      else:
        temp_dataset = temp_dataset.map(self.prepare_dataset, remove_columns=temp_dataset.column_names, num_proc=num_proc,batch_size=batch_size, writer_batch_size=num_writer)

      print(f"Info segment {index}")
      print(temp_dataset)

      avoid_OOM()
      temp_dataset.save_to_disk(save_folder / str(index), num_proc=num_proc)

      if cache_file_name_map is not None:
        shutil.rmtree(str(cache_file_name_map), ignore_errors=True)

      index+=1

    clear_audio_folder_cache(cache_path)


  def process_save_folder(self, data_folder, storage_folder, cache_file_name_map=None, batch_size=500, num_proc=10, num_writer=1000, keep_dataset_in_memory = False):
    need_to_process_folder = list_leaf_dirs(data_folder)

    for audio_folder in tqdm(need_to_process_folder):
      save_folder = pathlib.Path(str(audio_folder).replace(str(data_folder), str(storage_folder)))
      avoid_OOM()
      # cosider to mkdir with parent = True
      if not save_folder.exists():
        save_folder.mkdir(parents=True)
        print(f"Created {save_folder}")
        
        num_files = len(list(pathlib.Path(audio_folder).rglob('*.wav')))
        print(f"Processing {audio_folder} and store at {save_folder}")
        
        self.process_save(audio_folder, save_folder, cache_file_name_map, batch_size=batch_size, num_proc=int(min(num_files, num_proc)), num_writer = num_writer, keep_in_memory=keep_dataset_in_memory)
      else:
        print(f"Already processed{audio_folder} and store at {save_folder}")

      print(space)


  def process_save_folder_segment_mode(self, data_folder, storage_folder, cache_file_name_map=None, num_row=2500, batch_size=500, num_proc=10, num_writer=1000, keep_dataset_in_memory = False):
    need_to_process_folder = list_leaf_dirs(data_folder)

    for audio_folder in tqdm(need_to_process_folder):
      save_folder = pathlib.Path(str(audio_folder).replace(str(data_folder), str(storage_folder)))
      avoid_OOM()
      # cosider to mkdir with parent = True
      if not save_folder.exists():
        save_folder.mkdir(parents=True)
        print(f"Created {save_folder}")

      num_files = len(list(pathlib.Path(audio_folder).rglob('*.wav')))
      print(f"Processing {audio_folder} in segment mode and store at {save_folder}")

      self.process_save_segment_mode(audio_folder, save_folder, cache_file_name_map, batch_size=batch_size, num_proc=int(min(num_files, num_proc)), num_row=num_row, num_writer=num_writer, keep_in_memory=keep_dataset_in_memory)
        
      print(space)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--pv', help = 'processor version (tiny, base, small, medium, large)', choices=['tiny', 'base', 'small', 'medium', 'large'])
  parser.add_argument('--label', help = 'process with label or not', type = str, default='yes')
  parser.add_argument('--au', help='path to audio folder', type=str)
  parser.add_argument('--stro', help='path to processed data strogage folder (when task is 2, the format of the folder is different, it has 3 sub folders: train, validation, test)', type=str)
  parser.add_argument('--cap', help='audio folder cache path', type=str, default=None)
  parser.add_argument('--al', help='path to aligned folder, need to specific when task is 1', type=str)
  parser.add_argument('--task', help= 'task to do 0: format raw folder AFTER ALIGN, 1: process and store local data, 2: process and store local data in segment mode', type=str, default='')
  parser.add_argument('--capro', help='folder to store cache file batch for map function when process, must specify when task is 1 or 2 when your machine has low memory', type=str)
  parser.add_argument('--nr', help='num rum of each segment of in segment mode when segment the dataset (use in task 2)', type=int, default=1500)
  parser.add_argument('--nproc', help='num processor when multi processing', type=int, default=5)
  parser.add_argument('--wtbs', help='wrtiter batch size when map', type=int, default=500)

  args = parser.parse_args()

  processor_version = args.pv
  with_label = bool(args.label == 'yes')
  data_folder = pathlib.Path(args.au)
  aligned_folder = pathlib.Path(args.al) if args.al is not None else None
  strogage_folder = pathlib.Path(args.stro)
  task = args.task
  cache_path = pathlib.Path(args.cap) if args.cap is not None else None
  num_row = int(args.nr)
  cache_file_name_map_path = pathlib.Path(args.capro) if args.capro is not None else None
  num_proc = int(args.nproc)
  write_batch_size = int(args.wtbs)

  model_id =  "openai/whisper-" + processor_version
  processor = WhisperProcessor.from_pretrained(model_id, language="vi", task="transcribe")

  if with_label:
    print("with label")
  else:
    print("without label")

  if '0' in task:
    print(f"Format data at {data_folder}")
    formatter = local_data_formatter()
    formatter.format_folder(data_folder, aligned_folder,  delete_unaligned=True)
  
  if '1' in task:
    # load the processor and model
    print(f"Process data at {data_folder} and store at {strogage_folder}")
    data_process = local_dataset_processor(processor, with_label)
    data_process.process_save_folder(data_folder, strogage_folder, cache_file_name_map_path, num_proc=num_proc, num_writer=write_batch_size, keep_dataset_in_memory=True)
  
  if '2' in task:
    # load the processor and model
    print(f"Process data at {data_folder} and store at {strogage_folder} in segment mode")
    data_process = local_dataset_processor(processor, with_label)
    data_process.process_save_folder_segment_mode(data_folder, strogage_folder, cache_file_name_map_path, num_row=num_row, num_proc=num_proc, num_writer=write_batch_size)