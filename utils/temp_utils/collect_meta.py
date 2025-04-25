import pathlib
from tqdm import tqdm
import pandas as pd


def collect_meta(audio_path :pathlib.Path, transcript_path :pathlib.Path):
  dict = {'file_name' : [], 'transcription' : []}
  num = 0
  for i in tqdm(list(audio_path.rglob('*.wav'))):
    transcript_file = pathlib.Path(transcript_path / (i.stem + '.txt'))

    if not transcript_file.exists():
      num += 1
      continue
    
    with open(transcript_file, 'r', encoding='utf8') as r:
      transcript = r.read().lower().strip()

    dict['file_name'].append(str(i.name))
    dict['transcription'].append(transcript)
  
  print(f"Miss {num} files")
  df = pd.DataFrame.from_dict(dict)
  df.to_csv(audio_path / 'metadata.csv', index=False)

au_path = r""
trans_path = r""

collect_meta(au_path, trans_path)