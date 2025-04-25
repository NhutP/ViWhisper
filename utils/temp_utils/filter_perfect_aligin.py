import pathlib
import shutil
from tqdm import tqdm
import argparse


def delete_not_perfect(au_path :pathlib.Path, aligned_path: pathlib.Path):
  num_delete = 0
  for i in tqdm(list(aligned_path.rglob('*.txt'))):
    if '_1' not in i.parent.name:
      delete_file = pathlib.Path(str(i).replace(str(aligned_path), str(au_path))).parent.parent.parent / (i.stem + '.wav') 
      # print(delete_file)
      num_delete += 1
      if not delete_file.exists():
        continue
      delete_file.unlink()
  print(f"{num_delete} files was deleted")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('aufo', help='audio folder', type=str)
    parser.add_argument('alfo', help='algined folder if count perfect', type=str)

    args = parser.parse_args()

    delete_not_perfect(pathlib.Path(args.aufo), pathlib.Path(args.alfo))