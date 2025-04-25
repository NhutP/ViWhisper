import sys
sys.path.insert(0, r'../..')

import pandas as pd
import pathlib
from tqdm import tqdm
import argparse

from utils.prepare_data import list_leaf_dirs_with_extension, format_string



def get_all_delete_file_from_csv(csv_files: pathlib.Path):
  all_delete_files = []

  for i in csv_files.iterdir():
    all_delete_files += pd.read_csv(i)['file'].tolist()

  all_delete_files = list(set(all_delete_files))

  delete_file_grouped = {}

  for i in tqdm(all_delete_files):
    delete_file_grouped[str(pathlib.Path(i).parent)] = []

  for i in tqdm(all_delete_files):
    file = pathlib.Path(i)
    delete_file_grouped[str(file.parent)].append(file.name)

  return delete_file_grouped



def extract_and_save_metadata(leaf_original, delete_files):
  df = pd.read_csv(leaf_original / 'metadata.csv')

  file_name = df['file_name'].tolist()
  transcription = df['transcription'].tolist()

  delete_index = []
  # count = 0

  for i in range(len(file_name)):
    if str(file_name[i]) in delete_files:
      delete_index.append(i)
      (leaf_original / file_name[i]).unlink()
      # count += 1

  file_name = [file_name[i] for i in range(len(file_name)) if i not in delete_index]
  transcription = [transcription[i] for i in range(len(transcription)) if i not in delete_index]

  # return count
  pd.DataFrame.from_dict({'file_name' : file_name, 'transcription' : transcription}).to_csv(leaf_original / 'metadata.csv', encoding='utf8', index=False)



# def fix_folder(original_label_path, csv_path):
#   original_leaf_folders = list_leaf_dirs_with_extension(original_label_path, 'wav')

  
#   print("Get delete from csv files")
#   delete_info = get_all_delete_file_from_csv(csv_path)
#   folder_need_to_fix = delete_info.keys()
#   print("Start collecting")
#   count = 0

#   for i in tqdm(range(len(original_leaf_folders))):
#     count += extract_and_save_metadata(original_leaf_folders[i], delete_info[str(original_leaf_folders[i])])
  
#   print(count)


def fix_folder(csv_path):
 
  print("Get delete from csv files")
  delete_info = get_all_delete_file_from_csv(csv_path)
  folder_need_to_fix = list(delete_info.keys())
  print("Start collecting")
  # count = 0

  for i in tqdm(folder_need_to_fix):
    # count += extract_and_save_metadata(pathlib.Path(i), delete_info[i])
    extract_and_save_metadata(pathlib.Path(i), delete_info[i])




if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Filter fixed data")
  
  parser.add_argument('original_text_folder', type=str, help='original text folder')
  parser.add_argument('csv_path', type=str, help='csv folder, each csv must contains a "file" coulumn')
  

  args = parser.parse_args()

  original_folder = pathlib.Path(args.original_text_folder)
  csv_folder = pathlib.Path(args.csv_path)

  assert original_folder.exists()
  assert csv_folder.exists()

  fix_folder(csv_folder)