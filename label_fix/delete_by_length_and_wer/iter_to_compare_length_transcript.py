import sys
sys.path.insert(0, r'../..')

import pathlib
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from utils.prepare_data import list_leaf_dirs, format_string, list_leaf_dirs_with_extension
from utils.memory import avoid_OOM, clear_audio_folder_cache
import argparse
import pandas as pd
import evaluate

wer = evaluate.load('wer')

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
    concat_dataset = concatenate_datasets(datasets)
    return concat_dataset
  
  else:
    # avoid_OOM()
    dataset = load_dataset("audiofolder", data_dir=str(path), keep_in_memory = keep_dataset_in_memory, cache_dir=cache_dir)
    return dataset


def iter_and_check(dataset_dir: pathlib.Path, text_transcript_dir: pathlib.Path, df_path, cache_dir):
  dataset = load_dataset_recur(dataset_dir, cache_dir)
  print(dataset)

  eliminate_file = [] 
  mis_length = [] 
  scrip_df = []
  label_df = []

  print("Start checking")
  for i in tqdm(range(len(dataset))):
    audio_file = pathlib.Path(str(dataset[i]["audio"]["path"]))

    script_file_name = audio_file.stem + '.txt'
    script_file = pathlib.Path(str(audio_file).replace(str(dataset_dir), str(text_transcript_dir))).parent / script_file_name
    script_file = pathlib.Path(script_file)

    script = format_string(script_file.read_text(encoding='utf8'))
    assigned_label = format_string(dataset[i]["transcription"])

    # len_script = len(script.split())
    # len_assigned_label = len(assigned_label.split())

    wer_score = 100 * wer.compute(predictions=[script], references=[assigned_label])
    
    # if len_script != len_assigned_label:
    #   eliminate_file.append(audio_file)
    #   scrip_df.append(script)
    #   label_df.append(assigned_label)
    #   mis_length.append(len_script - len_assigned_label)

    # if wer_score > 0:
    #   eliminate_file.append(audio_file)
    #   scrip_df.append(script)
    #   label_df.append(assigned_label)
    #   mis_length.append(wer_score)
  
  
    eliminate_file.append(audio_file)
    scrip_df.append(script)
    label_df.append(assigned_label)
    mis_length.append(wer_score)

  # print(f"{len(eliminate_file)} file to be deleted")
  # pd.DataFrame.from_dict({'file' : eliminate_file, 'whisper_script' : scrip_df, 'assigned_label' : label_df, 'len_error' : mis_length}).to_csv(df_path, index=False)
  print(f"{len(eliminate_file)} file to be deleted")

  pd.DataFrame.from_dict({'file' : eliminate_file, 'whisper_script' : scrip_df, 'assigned_label' : label_df, 'wer_error' : mis_length}).sort_values(by="wer_error", ascending=False).to_csv(df_path, index=False)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Parse dataset path, script path, and cache directory.")

  parser.add_argument("dataset_path", type=str, help="The path to the dataset.")

  parser.add_argument("script_path", type=str, help="The path to the script.")

  parser.add_argument("cache_dir", type=str, help="The path to the cache directory.")

  parser.add_argument("df_path", type=str, help="The path csv file.")
  args = parser.parse_args()

  dataset_path = pathlib.Path(args.dataset_path)
  script_path = pathlib.Path(args.script_path)
  cache_dir = args.cache_dir
  df_path = args.df_path

  iter_and_check(dataset_path, script_path, df_path, cache_dir)