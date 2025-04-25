import pathlib
import pandas as pd
from tqdm import tqdm


path = pathlib.Path("/mnt/mmlab2024/datasets/data_fix/fixed_label/ytb_3_with_user_name")
excel_path = pathlib.Path(r"/mnt/mmlab2024/datasets/data_fix/count_label.xlsx")

def get_info(path :pathlib.Path):
  full_info = path.stem.split('_')
  name_with_ms = '_'.join(full_info[2:])
  return name_with_ms


def extract_info(name_with_ms):
  name_with_ms = name_with_ms.split('_')
  name = ' '.join(name_with_ms[0: len(name_with_ms) - 1])
  ms = name_with_ms[-1]
  return name, ms

contribute_count = {}

all_file = list(path.rglob('*.txt'))

for i in tqdm(all_file):
  name_with_ms = get_info(i)
  if name_with_ms not in contribute_count:
    contribute_count[name_with_ms] = 0
  contribute_count[name_with_ms] += 1

df = {'name' : [], 'mssv' : [], 'num' : []}

for i, j in contribute_count.items():
  name, ms = extract_info(i)
  df['name'].append(name)
  df['mssv'].append(ms)
  df['num'].append(int(j))

pd.DataFrame.from_dict(df).to_excel(excel_path, index=False)