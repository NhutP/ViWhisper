import pathlib
import pandas as pd
from tqdm import tqdm

def delete_untranscripted_files(path: pathlib.Path):
  num_deleted = 0

  df = pd.read_csv(str(path / 'metadata.csv'))
  file_name = df['file_name'].tolist()

  for i in tqdm(list(path.rglob('*.wav'))):
    if i.name not in file_name:
      num_deleted += 1
      i.unlink()
    
  print(f"Deleted {num_deleted} files")

path = pathlib.Path(r"")
delete_untranscripted_files(path)