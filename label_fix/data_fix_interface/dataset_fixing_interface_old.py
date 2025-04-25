import streamlit as st
import numpy as np
import librosa
from tqdm import tqdm
import pathlib 
from datasets import concatenate_datasets, load_dataset
from datetime import datetime
import time

global data_path, fixed_label_path

# data_path = pathlib.Path(r"C:\Users\quang\Desktop\testweb")
# fixed_label_path = pathlib.Path(r"C:\Users\quang\Desktop\fixedlabel")

data_path = pathlib.Path(r"/mnt/mmlab2024/datasets/Backup/final_dataset/splited_ytb_with_perfect")
fixed_label_path = pathlib.Path(r"/home/mmlab/s2t_history/fixed_label/ytb_3")

data_to_fix = lambda audio_file : pathlib.Path(str(audio_file).replace(str(data_path), str(fixed_label_path))).parent / (audio_file.stem + '.txt')

delete_chars = '!()-;:",.?@#$%&*()_[]'
def format_string(text :str):
  table = str.maketrans('', '', delete_chars)
  return text.translate(table).lower().strip()


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


@st.cache_resource()
def load_dataset_recur(path: pathlib.Path, keep_dataset_in_memory = False):
  leaf_dirs = list_leaf_dirs(path)

  if len(leaf_dirs) > 0:
    concat_dataset = None
    datasets = []
    print(f'Loading dataset at {str(path)}')
    for j in tqdm(leaf_dirs):
      datasets.append(load_dataset("audiofolder", data_dir=str(j), keep_in_memory = keep_dataset_in_memory)['train'])
      
    concat_dataset = concatenate_datasets(datasets)
    print(concat_dataset)
    return concat_dataset

  else:
    dataset = load_dataset("audiofolder", data_dir=str(path), keep_in_memory = keep_dataset_in_memory)['train']
    print(dataset)
    return dataset


def play_sample(array, sampling_rate=16000):
  st.audio(array, format='audio/wav', sample_rate=sampling_rate)


def get_random_sample_to_fix(dataset):
  count = 1
  current_index = np.random.randint(0, len(dataset))
  current_fixed_path = data_to_fix(pathlib.Path(dataset[current_index]['audio']['path']))

  while current_fixed_path.exists():
    if count == len(dataset):
      return None, None, None
    count += 1
    current_index = np.random.randint(0, len(dataset))
    current_fixed_path = data_to_fix(pathlib.Path(dataset[current_index]['audio']['path']))


  if count == len(dataset):
    return None, None, None
  
  return dataset[current_index]['audio']['array'], current_fixed_path, dataset[current_index]['transcription']

# def get_random_sample_to_fix(dataset):
#   count = 1
#   current_index = 0
#   current_fixed_path = data_to_fix(pathlib.Path(dataset[current_index]['audio']['path']))

#   while current_fixed_path.exists():
#     if count == len(dataset):
#       return None, None, None
#     count += 1
#     current_index += 1
#     current_fixed_path = data_to_fix(pathlib.Path(dataset[current_index]['audio']['path']))

#   if count == len(dataset):
#     return None, None, None
  
#   return dataset[current_index]['audio']['array'], current_fixed_path, dataset[current_index]['transcription']


def submit(fixed_text, path):
  pathlib.Path(path.parent).mkdir(exist_ok=True, parents=True)
  with open(path, 'w', encoding='utf8') as w:
    w.write(fixed_text)


# def render(dataset):
#   array, fixed_path, transciption = get_random_sample_to_fix(dataset)
#   print(fixed_path)
#   st.audio(array, sample_rate=16000)
#   fixed_text = st.text_area("Type what you listen", value=transciption)
#   submit_button = st.button('Submit')#, key=int(round(datetime.now().timestamp()))) 

#   if submit_button:
#     st.success("Thank you")
#     submit(fixed_text, fixed_path)
#     st.experimental_rerun()


def render(dataset):
  if 'current_sample' not in st.session_state:
    array, fixed_path, transcription = get_random_sample_to_fix(dataset)
    if array is None:
      st.success("The dataset has been completed, thank you")
      time.sleep(999999)
    st.session_state.current_sample = (array, fixed_path, transcription)
    st.session_state.fixed_text = transcription

  array, fixed_path, transcription = st.session_state.current_sample

  # Display the audio and text area
  st.audio(array, sample_rate=16000, autoplay=True)
  # fixed_text = st.text_area("Type what you listen", value=st.session_state.fixed_text, on_change=)
  # st.session_state.fixed_text = fixed_text

  # # Submit button
  # submit_button = st.button('Submit')

  # # If the submit button is clicked
  # if submit_button:
  #   st.success("Thank you")
  #   submit(fixed_text, fixed_path)
  #   del st.session_state.current_sample
  #   st.rerun()
  with st.form(key='fix_text_form'):
    fixed_text = st.text_area("Type what you listen", value=st.session_state.fixed_text)
    submit_button = st.form_submit_button('Submit')

    if submit_button:
      st.session_state.fixed_text = fixed_text
      st.success("Thank you")
      submit(fixed_text, fixed_path)
      del st.session_state.current_sample
      st.rerun()


def main():
  print('Check')
  print('Loading dataset')
  st.title("Vietnamese Speech to Text data fixing")
  st.warning("First time loading dataset, please wait ...")
  
  dataset = load_dataset_recur(data_path)
  
  render(dataset)

if __name__ == '__main__':
  main()