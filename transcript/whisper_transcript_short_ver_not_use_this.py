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
from whisper_train.load_data import whisper_data_loader


class whisper_transcripter:
  def __init__(self, path, pipe):
    self.loader = whisper_data_loader(pathlib.Path(path))
    self.pipe = pipe


  def writer(self, text, path_chunked_audio):
    with open(path_chunked_audio, 'w', encoding='utf8') as f:
      f.write(text)


  def transcript(self, source_folder, result_folder, total_num_thread, thread_index, batch_size=16, keep_in_memory=False, is_prepare_dir=False):

    dataset = self.loader.load_all(keep_dataset_in_memory=keep_in_memory)
    print("set format dataset")
    dataset = dataset.with_format('numpy')
    
    start_index = int(thread_index * len(dataset) / total_num_thread)
    end_index = int((thread_index + 1)* len(dataset) / total_num_thread) - 1

    dataset = dataset.select(list(range(start_index, end_index + 1)))

    des_path = [pathlib.Path(str(i).replace(str(source_folder), str(result_folder))).parent / (pathlib.Path(i).stem + '.txt') for i in dataset['path']]

    need_to_transcript_index = [i for i in range(len(dataset)) if not des_path[i].exists()]
    des_path = [des_path[i] for i in need_to_transcript_index]
    

    if is_prepare_dir:
      prepare_dir = set([i.parent for i in des_path])
      print("Prepare folder")
      for i in prepare_dir:
        pathlib.Path(i).mkdir(parents=True, exist_ok=True)


    print(f"There are {len(need_to_transcript_index)} need to be done")
    dataset = dataset.select(need_to_transcript_index)

    print(f'Start transcript thread {thread_index} / {total_num_thread} (from {start_index} to {end_index}), total {end_index - start_index + 1}')

    index = 0

    for script in self.pipe(KeyDataset(dataset, "array"), batch_size=batch_size):
      if des_path[index].exists():
        print(f"Stop beacause {des_path[index]} is alrerady existed")
        break
      self.writer(script['text'], des_path[index])
      index += 1

      if index % 1000 == 0:
        print(f"Finised {index}")


if __name__ == '__main__':
  assert torch.cuda.is_available()
  # device = 'cuda'

  parser = argparse.ArgumentParser()

  parser.add_argument('ver', help = 'model version (tiny, base, small, medium, large)', choices=['tiny', 'base', 'small', 'medium', 'large'])
  
  parser.add_argument('out', help='result transcript', type=str)
  parser.add_argument('bs', help = 'bacth size', type=int)
  parser.add_argument('ttth', help = 'total thread', type=int)
  parser.add_argument('threadindex', help = 'thread index', type=int)

  parser.add_argument('--savau', help = 'saved mapped dataset, which is the proceesed and contains .arrow files, must pass the source where the true .wav file store', type=str)
  parser.add_argument('--inau', help = 'source audio folder', type=str)
  
  parser.add_argument('--kime', help='keep in memory, "yes" if want to keep else do not use', type=str, default='no')
  parser.add_argument('--ppd', help='is_prepare_dir', type=str, default='no')
  parser.add_argument('--device', help='device to use (cuda:0, cuda:1, ...)', type=str, default='cuda')

  args = parser.parse_args()

  model_version = str(args.ver)
  batch_size = int(args.bs)
  input_audio = pathlib.Path(str(args.inau))
  output_transcript = pathlib.Path(str(args.out))
  total_thread = int(args.ttth)
  thread_index = int(args.threadindex)


  keep_dataset_in_memory = bool(args.kime == 'yes')
  saved_mapped_data_dir = pathlib.Path(args.savau)if args.savau is not None else None
  is_prepare_dir = bool(args.ppd == 'yes')

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

  tran = whisper_transcripter(saved_mapped_data_dir, pipe)
  
  tran.transcript(input_audio, output_transcript, total_thread, thread_index, batch_size=batch_size, is_prepare_dir=is_prepare_dir)