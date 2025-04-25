from tqdm import tqdm
import pathlib
import requests
import json
import wave
import numpy as np
import math
import random
import pandas as pd
import google.generativeai as genai
import os
import argparse
import time
import sys


def generate_random_binary_permutation(num_ones, total_length):
  """
  Generates a random binary permutation with a specified number of 1's.

  Parameters:
  num_ones (int): The number of 1's in the binary permutation.
  total_length (int): The total length of the binary permutation.

  Returns:
  list: A list representing the random binary permutation.
  """
  if num_ones > total_length:
      raise ValueError("Number of 1's cannot be greater than the total length.")

  # Create the initial list with the specified number of 1's and 0's
  binary_list = [1] * num_ones + [0] * (total_length - num_ones)
  
  # Shuffle the list to create a random permutation
  random.shuffle(binary_list)

  return binary_list


def get_wav_duration(file_path):
  # Open the .wav file
  with wave.open(file_path, 'rb') as wav_file:
      # Extract the number of frames
      n_frames = wav_file.getnframes()
      # Extract the sample rate
      frame_rate = wav_file.getframerate()
      
      # Calculate the duration
      duration = n_frames / float(frame_rate)
  return duration


def format_string(text :str):
  delete_chars = '!()-;:",.?@#$%&*()_[]'
  table = str.maketrans('', '', delete_chars)
  return text.translate(table).lower()


class speech_generator:
  def __init__(self, speech_api_key, vocab, appeard_words, generated_sentences, voices, url, min_unseen = 1, max_unseen = 3, min_length_for_random=5, max_length_for_random=15, gemini_api_key=None, unseen_words = None):
    self.url = url
    self.speech_api_key = speech_api_key
    self.vocab = vocab
    self.appear_words = appeard_words
    self.unseen_word = unseen_words
    self.voices = voices
    self.generated_sentences = generated_sentences
    self.speed_options = ['-1.5', '-1', '-0.5', '', '0.5', '1', '1.5']
    
    self.min_unseen = min_unseen
    self.max_unseen = max_unseen
    self.min_length_for_random = min_length_for_random
    self.max_length_for_random = max_length_for_random
    
    self.get_unseen_index = lambda size : np.random.randint(0, len(self.unseen_word), size).tolist()
    self.get_appeared_index = lambda size : np.random.randint(0, len(self.appear_words), size).tolist()
    self.generate_random_binary_permutation = generate_random_binary_permutation

    self.gemini_api_key = gemini_api_key

    if gemini_api_key is not None:
      genai.configure(api_key = self.gemini_api_key[0])

    if unseen_words is None:
      self.unseen_word = list(set(self.vocab).difference(self.appear_words))
      new_unseen = []

      for i in tqdm(self.unseen_word):
      
        for j in i.split():
          if j.strip() not in self.appear_words:
            new_unseen.append(i)
            break

      self.unseen_word = new_unseen

      with open(r"unseen.txt", 'w', encoding='utf8') as w:
        for i in self.unseen_word:
          w.write(i)
          w.write('\n')

    print(f"{len(self.unseen_word)} unseen words")

    

  def generate_next_text_random(self, length, unseen_proportion):
    num_of_unseen = math.ceil(length * unseen_proportion)
    num_of_appeared = length - num_of_unseen

    unseen_indexes = self.get_unseen_index(num_of_unseen)
    appeared_indexes = self.get_appeared_index(num_of_appeared)

    # 1 for unseen, 0 for appeared
    sentence_pattern = self.generate_random_binary_permutation(num_of_unseen, length)

    appeared_iter = 0
    unseen_iter = 0
    sum_iter = 0

    words = []

    while sum_iter < length:
      if sentence_pattern[sum_iter] == 1:
        words.append(self.unseen_word[unseen_indexes[unseen_iter]])
        unseen_iter += 1

      elif sentence_pattern[sum_iter] == 0:
        words.append(self.appear_words[appeared_indexes[appeared_iter]])
        appeared_iter += 1

      sum_iter += 1
    
    print(words)
    text = ' '.join(words)
    
    return text.strip()
      

  def generate_next_text_gemini(self, num_of_unseen):
  
    unseen_indexes = self.get_unseen_index(num_of_unseen)

    model = genai.GenerativeModel('gemini-pro')

    unseen_words = [self.unseen_word[i] for i in unseen_indexes]
    unseen_text = ', '.join(unseen_words)

    prompt = f'Đặt câu có nghĩa với {num_of_unseen} từ: {unseen_text}. Chỉ ghi câu trả lời, không cần giải thích'
    print('\n' + f"Prompt: {prompt}")
    response = model.generate_content(prompt)
    return response.text


  def save_audio_from_url(self, audio_url, write_path):
    audio_respone = requests.get(audio_url)

    if not (audio_respone.status_code == 200):
      print(audio_respone.status_code)
      print(url)
      return 0

    if audio_respone.status_code == 200:
      with open(write_path, "wb") as file:
        file.write(audio_respone.content)
    
    return 1


  def generate_speech_single_sentence(self, text, write_path, voice, speed, api_key):
    payload = text

    headers = {
    'api-key': api_key,
    'speed': str(speed),
    'voice': voice,
    'format' : 'wav'
    }

    response = requests.request('POST', self.url, data=payload.encode('utf-8'), headers=headers)
    audio_url = json.loads(response.text)["async"]
    time.sleep(10)
    audio_duration = self.save_audio_from_url(audio_url, write_path)
    if audio_duration == 0:
      print(headers)
    return audio_duration


  def generate_all_text(self, store_folder, strategy, start_name_index=0, num = 50000):
    start_name_index += 1
    store_folder = pathlib.Path(store_folder)

    print(f"{strategy}")
    
    if strategy == "random":
      generate_function = self.generate_next_text_random
    elif strategy == "gemini":
      generate_function = self.generate_next_text_gemini

    if strategy == "random":
      for _ in tqdm(range(num)):
        length = np.random.randint(self.min_length_for_random, self.max_length_for_random + 1)
        unseen_proportion = random.uniform(0.3, 0.6)

        while True:
          try:
            text = generate_function(length, unseen_proportion)
            break
          except Exception as e:
            print(f"Exception: {e}")
        file_name =  f"{(start_name_index):07n}.txt"
        with open(store_folder / file_name, 'w', encoding='utf8') as w:
          w.write(format_string(text))
        
        start_name_index += 1
    
    elif strategy == "gemini":
        API_index = 0
        for _ in tqdm(range(num)):
          unseen_num = np.random.randint(self.min_unseen, self.max_unseen + 1)

          while True:
            try:
              text = generate_function(unseen_num)
              break
            except Exception as e:
              print(f"Exception: {e}")
              if "429" in str(e):
                API_index = (API_index + 1) % len(self.gemini_api_key)
                genai.configure(api_key = self.gemini_api_key[API_index])
                print("change API key")
              
              if "Your default credentials were not found." in str(e):
                API_index = (API_index + 1) % len(self.gemini_api_key)
                genai.configure(api_key = self.gemini_api_key[API_index])
                print("change API key")

          file_name =  f"{(start_name_index):07n}.txt"
          with open(store_folder /  file_name, 'w', encoding='utf8') as w:
            w.write(format_string(text))
          
          start_name_index += 1
          time.sleep(5)
  

  def generate_speech(self, text_folder, result_store_folder):
    text_folder = pathlib.Path(text_folder)
    result_store_folder = pathlib.Path(result_store_folder)

    need_to_do_text = [j for j in text_folder.rglob('*.txt')]
    
    total_duration = 0
    for i in tqdm(need_to_do_text):
      result_file = pathlib.Path(str(i).replace(str(text_folder), str(result_store_folder))).parent / (i.stem + '.wav')
      
      if result_file.exists():
        continue
      
      print(f"Processing {result_file}")
      result_file.touch()

      voice = random.choice(self.voices)
      speed = random.choice(self.speed_options)

      with open(i, 'r', encoding='utf8') as r:
        text = r.read()
      
      current_duration = self.generate_speech_single_sentence(text, result_file, voice, speed, self.speech_api_key)

      while current_duration == 0:
        time.sleep(600)
        current_duration = self.generate_speech_single_sentence(text, result_file, voice, speed, self.speech_api_key)

      total_duration += current_duration
      
    return total_duration


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--inpa', type=str, help='The path to the input file or directory')
  parser.add_argument('--repa', type=str, help='The path to the result file or directory')
  parser.add_argument('--task', type=int, help='1: generate sentence, 2: generate speech')
  parser.add_argument('--sid', type=int, help='star index')
  parser.add_argument('--sapik', type=str, help='star index')
  
  args = parser.parse_args()

  task = args.task
  input_path = args.inpa
  output_path = args.repa
  start_index = args.sid
  speech_api_key = args.sapik


  url = 'https://api.fpt.ai/hmi/tts/v5'

  voices = ['banmai', 'leminh', 'thuminh', 'minhquang', 'myan', 'linhsan', 'giahuy', 'lannhi', 'ngoclam']

  appeared = pd.read_csv(r"appear.csv")['words'].tolist()

  with open(r"vocab.txt", 'r', encoding='utf8') as r:
    vocab = r.read().split('\n')
    vocab = list(set([format_string(i) for i in vocab]))


  with open(r"unseen.txt", 'r', encoding='utf8') as r:
    unseen_word = r.read().split('\n')
    if len(unseen_word) == 0:
      unseen_word = None
    else:
      unseen_word = list(set([format_string(i) for i in unseen_word]))
  
  with open("gemini_api_key.txt", 'r') as r:
    gemini_api_key = r.read().split('\n')

  speech_gene = speech_generator(speech_api_key, vocab, appeared, [], voices, url, gemini_api_key=gemini_api_key, unseen_words=unseen_word)

  if task == 1:
    speech_gene.generate_all_text(output_path, "gemini", start_index)
  elif task == 2:
    speech_gene.generate_speech(input_path, output_path)