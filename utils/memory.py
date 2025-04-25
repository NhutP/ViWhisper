import shutil
import psutil
import sys
import pathlib


def clear_audio_folder_cache(path):
  if path is None:
    print("Do not have audio folder cache")
    return
  
  shutil.rmtree(pathlib.Path(path), ignore_errors=True)
  print(f'Cleared audio folder cache at {path}')


def clear_load_data_cache(path):
  if path is None:
    print("Do not have load data cache")
    return
  
  shutil.rmtree(pathlib.Path(path), ignore_errors=True)
  print(f'Cleared load data folder cache at {path}')


def avoid_OOM(percent=50):
  '''
  Avoid out of memory when map dataset
  '''
  memo = psutil.virtual_memory()
  if int(memo[2]) > percent:
    print(f"Memory reached {memo[2]}%, terminate to avoid out of memory")
    sys.exit(0)
