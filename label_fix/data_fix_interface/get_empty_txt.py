import pathlib
from tqdm import tqdm
import argparse


def get_empty_text(path :pathlib.Path):
  for i in tqdm(list(path.rglob('*.txt'))):
    if len(i.read_text(encoding='utf8')) == 0:
      print(i)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Filter fixed data")
  
  parser.add_argument('folder', type=str)
  args = parser.parse_args()

  path = pathlib.Path(args.folder)
  get_empty_text(path)
