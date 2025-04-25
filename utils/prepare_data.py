import  pathlib
import re

delete_chars = '!()-;:",.?@#$%&*()_[]\n'
# table = str.maketrans('', '', delete_chars)
table = str.maketrans(delete_chars, ' ' * len(delete_chars))

def format_string(text :str):
  text = text.replace("<unk>"," ").replace("'", " ")
  return re.sub(r'\s+', ' ', text.translate(table).lower().strip())


# # this is for english mapping
# import pandas as pd
# mapping_table = pd.read_csv(r"/data1/mmlab/dataset/newcode/ViWHISPER/utils/speech_coding.csv")
# mapping_table = mapping_table.map(lambda x : str(x))
# mapping_table = mapping_table.map(format_string)
# mapping_table = mapping_table.sort_values(by='Phien_am', ascending=False, key=lambda col: col.str.len())

# vn_word = mapping_table['Phien_am'].tolist()
# multi_word = mapping_table['Word'].tolist()


# def format_multilingual(text :str):
#   text = format_string(text)
#   for i in range(len(vn_word)):
#     text = text.replace(vn_word[i], multi_word[i])
#   return text


def prepare_map_for_transcription(batch):
  audio = batch["audio"]
  batch['array'] = audio["array"]
  batch['path'] = audio['path']
  return batch


def prepare_map_for_eval_transcipt(batch):
  audio = batch["audio"]
  batch['array'] = audio["array"]
  batch['path'] = audio['path']
  batch['label'] = batch['transcription']
  return batch


def list_leaf_dirs(root_dir: pathlib.Path):
  root_dir = pathlib.Path(root_dir)
  leaf_dirs = []
  for path in root_dir.rglob("*"):
    if path.is_dir():
      is_leaf = True
      for i in path.iterdir():
        if i.is_dir():
          is_leaf = False
          break
      if is_leaf:
        leaf_dirs.append(path)
  
  return leaf_dirs


def list_leaf_dirs_with_extension(root_dir: pathlib.Path, extension):
  root_dir = pathlib.Path(root_dir)
  leaf_dirs = []
  for path in root_dir.rglob("*"):
    if path.is_dir():
      for i in path.glob('*.' + extension):
        leaf_dirs.append(path)
        break

  return leaf_dirs


def number_of_subdir_layer(path :pathlib.Path):
  all_subdirs = [i for i in path.iterdir() if i.is_dir()]

  if len(all_subdirs) == 0:
    return 1
  else:
    return 1 + max([number_of_subdir_layer(i) for i in all_subdirs])

