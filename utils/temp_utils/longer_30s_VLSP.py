import pathlib
import librosa
from tqdm import tqdm
import shutil

path_au = pathlib.Path(r"/mnt/mmlab2024/datasets/VNSTT/formated_VLSP/data/VLSP_shorter_30s")
des_au = pathlib.Path(r"/mnt/mmlab2024/datasets/VNSTT/formated_VLSP/data/VLSP_longer_30s")

num = 0

for i in tqdm(list(path_au.rglob('*.wav'))):
  x, sr = librosa.load(i, sr = None)
  if x.shape[0] / sr > 30:
    shutil.copy2(str(path_au / i.name), str(des_au / i.name))
    num+=1

print(f"Total {num} files > 30s") 