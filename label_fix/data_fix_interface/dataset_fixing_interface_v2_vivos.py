import streamlit as st
import numpy as np
import librosa
from tqdm import tqdm
import pathlib 
from datasets import concatenate_datasets, load_dataset
from datetime import datetime
import time
import pandas as pd

global data_path, fixed_label_path


# data_path = pathlib.Path(r"C:\Users\quang\Desktop\testweb")
# fixed_label_path = pathlib.Path(r"C:\Users\quang\Desktop\fixedlabel")

# data_path = pathlib.Path(r"/mnt/mmlab2024/datasets/Backup/final_dataset/splited_ytb_with_perfect")
# fixed_label_path = pathlib.Path(r"/home/mmlab/s2t_history/fixed_label/ytb_3")

data_path = pathlib.Path(r"/mnt/mmlab2024/datasets/VNSTT/vivos_raw/test")
fixed_label_path = pathlib.Path(r"/home/mmlab/s2t_history/fixed_label/vivos/test")

delete_chars = '!()-;:",.?@#$%&*()_[]'
def format_string(text :str):
  table = str.maketrans('', '', delete_chars)
  return text.translate(table).lower().strip()


data_to_fix = lambda audio_file : pathlib.Path(str(audio_file).replace(str(data_path), str(fixed_label_path))).parent / (audio_file.stem + '.txt')


@st.cache_resource
def list_leaf_dirs(root_dir: pathlib.Path):
  extract_text = lambda df, audio_name : df[df['file_name'] == audio_name]['transcription'].item()
  audio_list = list(root_dir.rglob('*.wav'))
  print(len(audio_list))
  audio_list = [i for i in audio_list if not data_to_fix(i).exists()]
  print(len(audio_list))
  script = [extract_text(pd.read_csv(audio_file.parent / 'metadata.csv'), audio_file.name) for audio_file in tqdm(audio_list)]
  return audio_list, script


# @st.cache_resource
# def list_leaf_dirs(root_dir: pathlib.Path):
#   with open('vivos_unfix.txt', 'r', encoding='utf8') as r:
#     vivos_unfix = r.read().split('\n')

#   extract_text = lambda df, audio_name : df[df['file_name'] == audio_name]['transcription'].item()
#   audio_list = list(root_dir.rglob('*.wav'))
#   print(len(audio_list))
#   audio_list = [i for i in tqdm(audio_list) if not data_to_fix(i).exists() and extract_text(pd.read_csv(i.parent / 'metadata.csv'), i.name) in vivos_unfix]
#   print(len(audio_list))
#   script = [extract_text(pd.read_csv(audio_file.parent / 'metadata.csv'), audio_file.name) for audio_file in tqdm(audio_list)]
#   return audio_list, script


# @st.cache_resource
# def list_leaf_dirs(root_dir: pathlib.Path):
#   import evaluate
#   trans_path = pathlib.Path(r"/mnt/mmlab2024/datasets/VNSTT/script_vivos")

#   extract_text = lambda df, audio_name : df[df['file_name'] == audio_name]['transcription'].item()
#   audio_list = list(root_dir.rglob('*.wav'))
#   print(len(audio_list))
#   audio_list = [i for i in audio_list if not data_to_fix(i).exists()]
#   print(len(audio_list))
#   script = [extract_text(pd.read_csv(audio_file.parent / 'metadata.csv'), audio_file.name) for audio_file in tqdm(audio_list)]

#   trans_file = [trans_path / (i.stem + '.txt')  for i in audio_list]
#   trans = []

#   print("Read transcript")
#   for j in tqdm(trans_file):
#     with open(j, 'r', encoding='utf8') as r:
#       trans.append(format_string(r.read()))

#   print('Compute score')
#   wer = evaluate.load("wer")
#   score = [100 * wer.compute(predictions=[trans[i]], references=[script[i]]) for i in tqdm(range(len(script)))]
#   df = pd.DataFrame.from_dict({'audio_list' : audio_list, 'script' : script, 'score' : score}).sort_values(by='score', ascending=False)

#   print(trans[0])
#   print(script[0])
#   print(score[0])

#   print('--------------------------')

  

#   audio_list = df['audio_list'].tolist()
#   script = df['script'].tolist()
#   score = df['score'].tolist()

#   print(trans[11])
#   print(script[11])
#   print(score[11])

#   index_wer_positive = [i for i in range(len(score)) if score[i] > 0]

#   audio_list = [audio_list[i] for i in index_wer_positive] 
#   script = [script[i] for i in index_wer_positive]

#   print(len(audio_list))
#   return audio_list, script

def play_sample(array, sampling_rate=16000):
  st.audio(array, format='audio/wav', sample_rate=sampling_rate)


def get_random_sample_to_fix(audio_list, text_script):
  if len(audio_list) == 0:
    return None, None, None
  count = 0
  if len(audio_list) < 0:
    current_index = 0
  else:
    current_index = np.random.randint(0, len(audio_list))
  current_fixed_path = data_to_fix(audio_list[current_index])

  while current_fixed_path.exists():
    if count == 2 * len(audio_list):
      return None, None, None
    count += 1
    if len(audio_list) < 0:
      current_index = min(current_index + 1, len(audio_list) - 1)
    else:
      current_index = np.random.randint(0, len(audio_list))
    current_fixed_path = data_to_fix(audio_list[current_index])


  if count == 2 * len(audio_list):
    return None, None, None
  
  array, _ = librosa.load(audio_list[current_index], sr = 16000)

  return array, current_fixed_path, text_script[current_index]


def submit(fixed_text, path):
  pathlib.Path(path.parent).mkdir(exist_ok=True, parents=True)
  with open(path, 'w', encoding='utf8') as w:
    w.write(fixed_text)


def render(audio_list, script):
  if 'current_sample' not in st.session_state:
    array, fixed_path, transcription = get_random_sample_to_fix(audio_list, script)
    if array is None:
      st.success("The dataset has been completed, thank you")
      time.sleep(999999)
    st.session_state.current_sample = (array, fixed_path, transcription)
    st.session_state.fixed_text = transcription

  array, fixed_path, transcription = st.session_state.current_sample

  # Display the audio and text area
  st.audio(array, sample_rate=16000, autoplay=True)

  with st.form(key='fix_text_form'):
    fixed_text = st.text_area("Help us check the transcript of the audio above", value=st.session_state.fixed_text)
    submit_button = st.form_submit_button('Submit')

    if submit_button:
      st.session_state.fixed_text = fixed_text
      st.success("Thank you")
      submit(fixed_text, fixed_path)
      del st.session_state.current_sample
      st.rerun()


def main():
  st.title("Vietnamese Speech2Text labeling")

  if len(list(fixed_label_path.iterdir())) == 0:
    st.warning("First time loading dataset, please wait ...")

  audio_list, script = list_leaf_dirs(data_path)
  
  render(audio_list, script)

if __name__ == '__main__':
  main()