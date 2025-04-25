import pathlib
import argparse
from underthesea import word_tokenize
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

def get_vocab(path, output_path):
  path = pathlib.Path(path)
  output_path = pathlib.Path(output_path)

  words = []

  for i in tqdm(list(path.rglob("*.txt"))):
    with open(i, 'r', encoding='utf8') as r:
      res_text = word_tokenize(r.read().strip())
      words += res_text
  
  word_info = {}
  words_set = set(words)
  single_word_list = []

  print("single words")
  for j in tqdm(words_set):
    single_word_list += j.split()

  single_word_list = set(single_word_list)

  print("initializing single words")
  for j in single_word_list:
    word_info[j] = 0

  print("initializing words")
  for j in words_set:
    word_info[j] = 0

  print('counting')
  for j in tqdm(words):
    word_info[j] += 1
    for k in j.split():
      word_info[k] += 1

  word_info_df = {'words' : list(word_info.keys()), 'num_appear' : list(word_info.values())}

  df = pd.DataFrame.from_dict(word_info_df)
  df.to_csv(output_path)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('input_path', type=str, help="The path to the input folder, this contains .txt files")
  parser.add_argument('output_path', type=str, help="The path to the output file (*.csv)")

  args = parser.parse_args()

  input_path = args.input_path
  output_path = args.output_path

  get_vocab(input_path, output_path)