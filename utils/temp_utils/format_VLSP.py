import pathlib
import pandas as pd
from tqdm import tqdm


def format(path: pathlib.Path):
  dict = {'file_name' : [], 'transcription' : []}
  num = 0
  for i in tqdm(list(path.rglob('*.wav'))):
    transcript_file = pathlib.Path(i.parent / (i.stem + '.txt'))

    if not transcript_file.exists():
      num += 1
      continue
    
    with open(transcript_file, 'r', encoding='utf8') as r:
      transcript = r.read().lower().strip()

    dict['file_name'].append(str(i.name))
    dict['transcription'].append(transcript)
  
  print(f"Miss {num} files")
  df = pd.DataFrame.from_dict(dict)
  df.to_csv(path / 'metadata.csv', index=False)

path = pathlib.Path(r"")

format(path)