# use heuristics to filter out some confident sample
import sys
sys.path.insert(0, r'../..')

import pathlib
from tqdm import tqdm
import evaluate
import argparse
import pandas as pd

from utils.prepare_data import list_leaf_dirs_with_extension, format_string


class dataset_fixing_filter:
  def __init__(self, min_complete_file = 3, min_complete_ratio = 0.1, wer_threshold = 8, min_perfect_match = 5):
    self.min_complete_file = min_complete_file
    self.min_complete_ratio = min_complete_ratio
    self.wer_threshold = wer_threshold
    self.min_perfect_match = min_perfect_match

    self.wer = evaluate.load("wer")


  def is_skip(self, original_label_leaf_folder_path :pathlib.Path, fixed_label_leaf_folder_path :pathlib.Path):
    original_df = pd.read_csv(original_label_leaf_folder_path / 'metadata.csv').sort_values(by="file_name")
    
    text_for_wer = {'original' : [], 'fixed' : []}
    
    original_audio_name = original_df['file_name'].tolist()
    total_file_in_original = len(original_audio_name)

    # mapping_fixed_txt = [pathlib.Path(str(i).replace(str(original_label_leaf_folder_path), str(fixed_label_leaf_folder_path))).parent / (original_audio_name.split('.')[0] + '.txt') for i in original_audio_name]

    mapping_fixed_txt = [pathlib.Path(fixed_label_leaf_folder_path) / (i.split('.')[0] + '.txt') for i in original_audio_name]

    fixed_txt = [i if i.exists() else None for i in mapping_fixed_txt]
  
    perfect_match_count = 0

    for i in range(total_file_in_original):
      if fixed_txt[i] is None:
        continue
      ori_text = format_string(original_df['transcription'][i])
      fixed_text = format_string(fixed_txt[i].read_text(encoding='utf8'))
      
      if text_for_wer['original'] == fixed_text:
        perfect_match_count += 1

      text_for_wer['original'].append(ori_text)
      text_for_wer['fixed'].append(fixed_text)
    
    # print(len(text_for_wer['fixed']))
    if len(text_for_wer['fixed']) < (total_file_in_original * self.min_complete_ratio):
      return False
    
    if len(text_for_wer['fixed']) < self.min_complete_file:
      return False

    if perfect_match_count < min(self.min_perfect_match, len(text_for_wer['original'])):
      return False

    score = 100 * self.wer.compute(predictions= text_for_wer['fixed'], references=text_for_wer['original'])

    if score > self.wer_threshold:
      return False
    
    return True


  def filter(self, original_label_path, fixed_label_path):
    original_leaf_folders = list_leaf_dirs_with_extension(original_label_path, 'wav')

    mapping_fixed_leaf_folder = [pathlib.Path(str(i).replace(str(original_label_path), str(fixed_label_path))) for i in original_leaf_folders]

    exist_fixed_leaf_folder = [i if i.exists() else None for i in mapping_fixed_leaf_folder]
    
    print("Start filtering")
    for i in tqdm(range(len(exist_fixed_leaf_folder))):
      if exist_fixed_leaf_folder[i] is None:
        continue
      
      if self.is_skip(original_leaf_folders[i], exist_fixed_leaf_folder[i]):
        ## delete code here
        print(f"Eliminate {original_leaf_folders[i]} ------ {exist_fixed_leaf_folder[i]}\n")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Filter fixed data")
  
  parser.add_argument('original_text_folder', type=str, help='original text folder')
  parser.add_argument('fixed_text_folder', type=str, help='fixed text folder')
  

  args = parser.parse_args()

  original_text_folder = pathlib.Path(args.original_text_folder)
  fixed_text_folder = pathlib.Path(args.fixed_text_folder)

  assert original_text_folder.exists()
  assert fixed_text_folder.exists()

  filter_fixed_data = dataset_fixing_filter()
  filter_fixed_data.filter(original_text_folder, fixed_text_folder)