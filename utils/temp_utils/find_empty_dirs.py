from pathlib import Path
import argparse
from tqdm import tqdm
import shutil
import pathlib

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

def find_empty_folders(directory):
  dir_path = Path(directory)
  empty_folders = []

  if not dir_path.is_dir():
      raise ValueError(f"{directory} is not a valid directory")

  # for folder in tqdm(list(dir_path.rglob('*'))):
  # for folder in tqdm(list_leaf_dirs(dir_path)):
  #   count = 0
  #   if folder.is_dir() and len(list(folder.iterdir())) < 2:
  #       empty_folders.append(folder)

  for folder in tqdm(list_leaf_dirs(dir_path)):
    if folder.is_dir() and len(list(folder.iterdir())) < 3:
      empty_folders.append(folder)

  return empty_folders

def main():
  parser = argparse.ArgumentParser(description="Find all empty folders in a directory recursively.")
  parser.add_argument("directory", type=str, help="The path to the directory to search.")
  args = parser.parse_args()

  empty_folders = find_empty_folders(args.directory)
  
  for folder in empty_folders:
    # shutil.rmtree(str(folder))
    print(folder)


if __name__ == "__main__":
  main()
