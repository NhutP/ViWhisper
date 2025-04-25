import pathlib
from tqdm import tqdm
import shutil
import argparse


def copy_all_csv(src_path :pathlib.Path, des_path: pathlib.Path):
  print("Start copy")
  for i in tqdm(list(src_path.rglob('*.csv'))):
    pathlib.Path(str(i.parent).replace(str(src_path), str(des_path))).mkdir(parents=True, exist_ok=True)
    shutil.copy2(i, str(i).replace(str(src_path), str(des_path)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Filter fixed data")
  
  parser.add_argument('original_folder', type=str, help='original folder')
  parser.add_argument('fixed_folder', type=str, help='destination folder')
  

  args = parser.parse_args()

  original_folder = pathlib.Path(args.original_folder)
  fixed_text_folder = pathlib.Path(args.fixed_folder)

  assert original_folder.exists()
  assert fixed_text_folder.exists()

  copy_all_csv(original_folder, fixed_text_folder)