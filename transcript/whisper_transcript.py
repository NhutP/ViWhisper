import sys
sys.path.insert(0, r'..')

from datasets import load_dataset, Audio, load_from_disk
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
import torch
import pathlib
import argparse
from utils.prepare_data import prepare_map_for_transcription, format_string, list_leaf_dirs, list_leaf_dirs_with_extension
from utils.memory import clear_audio_folder_cache


class transcripter:
  def __init__(self, pipe):
    self.pipe = pipe
  

  def get_writer(self, mode = 'all'):
    def write_chunk_only(text, path_chunked_audio):
      with open(path_chunked_audio, 'w', encoding='utf8') as f:
        f.write(text)
    
    def write_all(text, path_chunked_audio, path_raw):
      with open(path_chunked_audio, 'w', encoding='utf8') as f:
        f.write(text)
      
      with open(path_raw, 'a', encoding='utf8') as f:
        f.write(text + '\n')

    if mode == 'all':
      return write_all
    elif mode == "chunk":
      return write_chunk_only
    

  def load_only_audio(self, path, cache_file_name_map=None, num_proc=5, write_batch_size=500, keep_in_memory = False, load_num_proc = None, save_mapped_dataset = None):
    is_audiofolder = False
    for i in pathlib.Path(path).rglob('*'):
      if i.is_file() and '.wav' in i.name:
        print("Load with audio folder")
        is_audiofolder = True
        dataset = load_dataset("audiofolder", data_dir=str(path), drop_metadata=True, drop_labels=True, keep_in_memory=keep_in_memory, num_proc= load_num_proc)
        dataset = dataset['train']

        if cache_file_name_map is not None:
          dataset = dataset.map(prepare_map_for_transcription, remove_columns=dataset.column_names, num_proc=num_proc, cache_file_name=str(cache_file_name_map / 'temp.arrow'), writer_batch_size = write_batch_size, keep_in_memory=keep_in_memory)
        else:
          dataset = dataset.map(prepare_map_for_transcription, remove_columns=dataset.column_names, num_proc=num_proc, writer_batch_size = write_batch_size, keep_in_memory=keep_in_memory)

          dataset.set_format('np')

          if save_mapped_dataset is not None and is_audiofolder:
            print(f"Save dataset at {save_mapped_dataset}")
            dataset.save_to_disk(save_mapped_dataset, num_proc=load_num_proc)

        break

    if not is_audiofolder:
      print('Load saved data')
      dataset = load_from_disk(str(path), keep_in_memory=keep_in_memory)
      dataset.set_format('np')
      
    return dataset
  

  def transcript(self, source_folder, result_folder, total_num_thread, thread_index, cache_file_name_map, saved_datadir =None, batch_size=16, raw_folder=None, num_proc=5, write_batch_size=500, keep_dataset_in_memory = False, audiofolder_cache_file = None, load_num_proc = None, save_mapped_dataset = None):

    print('Prepare result folder')
    leaf_dirs = list_leaf_dirs_with_extension(source_folder, 'wav')
    prepare_dir = [pathlib.Path(str(i).replace(str(source_folder), str(result_folder))) for i in leaf_dirs]

    for i in prepare_dir:
      i.mkdir(exist_ok=True, parents=True)

    
    if saved_datadir is not None:
      print('Loading saved dataset')
      dataset = self.load_only_audio(saved_datadir, cache_file_name_map=cache_file_name_map, num_proc=num_proc, write_batch_size=write_batch_size, keep_in_memory=keep_dataset_in_memory, load_num_proc=load_num_proc, save_mapped_dataset=save_mapped_dataset)
    else:
      print('Loading from scratch')
      dataset = self.load_only_audio(source_folder, cache_file_name_map=cache_file_name_map, num_proc=num_proc, write_batch_size=write_batch_size, keep_in_memory=keep_dataset_in_memory, load_num_proc=load_num_proc, save_mapped_dataset=save_mapped_dataset)

    start_index = int(thread_index * len(dataset) / total_num_thread)
    end_index = int((thread_index + 1)* len(dataset) / total_num_thread) - 1

    dataset = dataset.select(list(range(start_index, end_index + 1)))

    des_path = [pathlib.Path(str(i).replace(str(source_folder), str(result_folder))).parent / (pathlib.Path(i).stem + '.txt') for i in dataset['path']]

    need_to_transcript_index = [i for i in range(len(dataset)) if not des_path[i].exists()]
    des_path = [des_path[i] for i in need_to_transcript_index]

    print(f"There are {len(need_to_transcript_index)} need to be done")
    dataset = dataset.select(need_to_transcript_index)

    print(f'Start transcript thread {thread_index} / {total_num_thread} (from {start_index} to {end_index}), total {end_index - start_index + 1}')
    index = 0

    if raw_folder is not None:
      writer = self.get_writer(mode='all')

      for script in self.pipe(KeyDataset(dataset, "array"), batch_size=batch_size):
        if des_path[index].exists():
          print(f"Stop beacause {des_path[index]} is alrerady existed")
          break
        writer(format_string(script['text']), des_path[index], raw_folder / (str(des_path[index].parent.stem) + '.txt'))
        index += 1
    
    else:
      writer = self.get_writer(mode='chunk')

      for script in self.pipe(KeyDataset(dataset, "array"), batch_size=batch_size):
        if des_path[index].exists():
          print(f"Stop beacause {des_path[index]} is alrerady existed")
          break
        writer(format_string(script['text']), des_path[index])
        index += 1

    if audiofolder_cache_file is not None:
      clear_audio_folder_cache(audiofolder_cache_file)


if __name__ == '__main__':
  assert torch.cuda.is_available()
  # device = 'cuda'


  parser = argparse.ArgumentParser()

  parser.add_argument('ver', help = 'model version (tiny, base, small, medium, large)', choices=['tiny', 'base', 'small', 'medium', 'large'])
  
  parser.add_argument('out', help='result transcript', type=str)
  parser.add_argument('bs', help = 'bacth size', type=int)
  parser.add_argument('ttth', help = 'total thread', type=int)
  parser.add_argument('threadindex', help = 'thread index', type=int)

  parser.add_argument('--savau', help = 'saved dataset, which is the proceesed and contains .arrow files, must pass the source where the true .wav file store', type=str)
  parser.add_argument('--inau', help = 'source audio folder', type=str)
  parser.add_argument('--aufolcache', help='audio folder cache file', type=str)
  parser.add_argument('--samap', help='save mapped dataset, if spcified and the dataset has not been processed, save the dataset after mapping', type=str)
  parser.add_argument('--capro', help='folder to store cache file batch for map function when process, must specify have low RAM', type=str)
  parser.add_argument('--ratran', help='raw transcription', type=str)
  parser.add_argument('--nproc', help='num processor when multi processing map data, default is none, which means dont use multi processing', type=int)
  parser.add_argument('--loadnproc', help='num processor when multi processing load data, default is none, which means dont use multi processing', type=int)
  parser.add_argument('--wtbs', help='wrtiter batch size when map', type=int, default=5000)
  parser.add_argument('--kime', help='keep in memory, "yes" if want to keep else do not use', type=str, default='no')
  parser.add_argument('--device', help='device to use (cuda:0, cuda:1, ...)', type=str, default='cuda')

  args = parser.parse_args()

  model_version = str(args.ver)
  batch_size = int(args.bs)
  input_audio = pathlib.Path(str(args.inau))
  output_transcript = pathlib.Path(str(args.out))
  total_thread = int(args.ttth)
  thread_index = int(args.threadindex)

  raw_transcript = pathlib.Path(args.ratran) if args.ratran is not None else None

  cache_file_name_map_path = pathlib.Path(args.capro) if args.capro is not None else None
  num_proc = int(args.nproc) if args.nproc is not None else None
  load_num_proc = int(args.loadnproc) if args.loadnproc is not None else None
  write_batch_size = int(args.wtbs)
  keep_dataset_in_memory = bool(args.kime == 'yes')
  audiofolder_cache_file =  pathlib.Path(args.aufolcache) if args.aufolcache is not None else None
  save_mapped_dataset = pathlib.Path(args.samap) if args.samap is not None else None
  saved_data_dir = pathlib.Path(args.savau)if args.savau is not None else None

  device = str(args.device)

  print(f"Device: {device}")


  pipe = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-" + model_version,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=batch_size,
    return_timestamps=False,
    device=device
  ) 

  tran = transcripter(pipe)
  
  tran.transcript(input_audio, output_transcript, total_thread, thread_index, cache_file_name_map_path, saved_datadir=saved_data_dir, batch_size=batch_size, raw_folder=raw_transcript, num_proc=num_proc, write_batch_size=write_batch_size, keep_dataset_in_memory=keep_dataset_in_memory, audiofolder_cache_file=audiofolder_cache_file, load_num_proc=load_num_proc, save_mapped_dataset=save_mapped_dataset)