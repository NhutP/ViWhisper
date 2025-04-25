import pathlib
import pandas as pd
from tqdm import tqdm
import argparse

if __name__ == '__main__':
  # Create the parser
  parser = argparse.ArgumentParser(description="Process a file path.")

  # Add the path argument
  parser.add_argument('path', type=str, help='The path to the file')

  # Parse the arguments
  args = parser.parse_args()

  path = pathlib.Path(args.path)

  num_words = 0
  total_sen = 0

  for i in tqdm(list(path.rglob('*.csv'))):
    df = pd.read_csv(i, encoding='utf8')
    for j in df['transcription'].tolist():
      num_words += len(j.split())
      total_sen += 1

  print(num_words)
  print(total_sen)
  print(num_words / total_sen)