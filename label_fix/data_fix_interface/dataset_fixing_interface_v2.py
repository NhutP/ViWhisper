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


# data_path = pathlib.Path(r"C:\Users\quang\Desktop\testweb\day01\@mingaidiu1998_1")
# fixed_label_path = pathlib.Path(r"C:\Users\quang\Desktop\fixedlabel")
# fixed_label_path_with_username = pathlib.Path(r"C:\Users\quang\Desktop\fixedlabel_username")


data_path = pathlib.Path(r"/mnt/mmlab2024/datasets/Backup/final_dataset/splited_ytb_with_perfect")
fixed_label_path = pathlib.Path(r"/home/mmlab/s2t_history/fixed_label/ytb_3_new")
fixed_label_path_with_username = pathlib.Path(r"/home/mmlab/s2t_history/fixed_label/ytb_3_with_user_name")


delete_chars = '!()-;:",.?@#$%&*()_[]'
def format_string(text :str):
  table = str.maketrans('', '', delete_chars)
  return text.translate(table).lower().strip()


data_to_fix = lambda audio_file : pathlib.Path(str(audio_file).replace(str(data_path), str(fixed_label_path))).parent / (audio_file.stem + '.txt')

add_username = lambda original_txt_file, name_words, mssv = None : pathlib.Path(str(original_txt_file.parent).replace(str(fixed_label_path), str(fixed_label_path_with_username))) / (original_txt_file.stem + '_' + '_'.join(name_words) + (('_' + mssv) if mssv is not None else '') + '.txt')


@st.cache_resource
def list_leaf_dirs(root_dir: pathlib.Path):
  extract_text = lambda df, audio_name : df[df['file_name'] == audio_name]['transcription'].item()
  audio_list = list(root_dir.rglob('*.wav'))
  print(len(audio_list))
  audio_list = [i for i in audio_list if not data_to_fix(i).exists()]
  print(len(audio_list))
  script = [extract_text(pd.read_csv(audio_file.parent / 'metadata.csv'), audio_file.name) for audio_file in tqdm(audio_list)]
  return audio_list, script


def play_sample(array, sampling_rate=16000):
  st.audio(array, format='audio/wav', sample_rate=sampling_rate)


# def get_random_sample_to_fix(audio_list, text_script):
#   count = 0
#   current_index = np.random.randint(0, len(audio_list))
#   current_fixed_path = data_to_fix(audio_list[current_index])

#   while current_fixed_path.exists():
#     if count == 2*len(audio_list):
#       return None, None, None
#     count += 1
#     current_index = np.random.randint(0, len(audio_list))
#     current_fixed_path = data_to_fix(audio_list[current_index])


#   if count == 2*len(audio_list):
#     return None, None, None
  
#   array, _ = librosa.load(audio_list[current_index], sr = 16000)

#   return array, current_fixed_path, text_script[current_index]

def get_random_sample_to_fix(audio_list, text_script):
  if len(audio_list) == 0:
    return None, None, None
  count = 0
  if len(audio_list) < 100:
    current_index = 0
  else:
    current_index = np.random.randint(0, len(audio_list))
  current_fixed_path = data_to_fix(audio_list[current_index])

  while current_fixed_path.exists():
    if count == 2 * len(audio_list):
      return None, None, None
    count += 1
    if len(audio_list) < 100:
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

  username_txt = add_username(path, st.session_state.username.split(), st.session_state.mssv)
  username_txt.parent.mkdir(parents=True, exist_ok=True)

  with open(username_txt, 'w', encoding='utf8') as w:
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
      submit(format_string(fixed_text), fixed_path)
      del st.session_state.current_sample
      st.rerun()

  if st.button('Reload'):
    del st.session_state.current_sample
    st.rerun()

def main():
  st.title("Vietnamese Speech2Text labeling")
  
    # Initialize session state if not present
  if 'username' not in st.session_state:
      st.session_state.username = ''
  if 'mssv' not in st.session_state:
      st.session_state.mssv = ''
  if 'submitted' not in st.session_state:
      st.session_state.submitted = False

  # Display the input fields and submit button only if the form has not been submitted
  if not st.session_state.submitted:
      st.session_state.username = format_string(st.text_input("Please enter your username:"))
      st.session_state.mssv = format_string(st.text_input("Please enter your MSSV, if you do not have, type 0"))

      # Add a submit button
      if st.button("Submit"):
          if st.session_state.username and st.session_state.mssv:
              st.session_state.username = format_string(st.session_state.username)
              st.session_state.mssv = format_string(st.session_state.mssv)
              st.session_state.submitted = True
              st.rerun()
          else:
              st.error("Both username and MSSV cannot be empty. Please enter valid values.")
      st.stop()
  # After submission, display the greeting and MSSV, and hide the input fields
  if st.session_state.submitted:
      st.write(f"Hello, {st.session_state.username.upper()}!")
      st.write(f"MSSV: {st.session_state.mssv}")
      # Add the rest of your app's logic here
  
  # if not st.session_state.username:
  #     st.stop()

  if len(list(fixed_label_path.iterdir())) == 0:
    st.warning("First time loading dataset, please wait ...")
  st.markdown("[Please add to this when you listen to a word in foreign language](https://docs.google.com/spreadsheets/d/1NYyA-UsqhIqZ6U6K5WuceoAbUaeshxYqO_kJxC7CLAk/edit?usp=sharing)")

  audio_list, script = list_leaf_dirs(data_path)
  
  render(audio_list, script)

if __name__ == '__main__':
  main()