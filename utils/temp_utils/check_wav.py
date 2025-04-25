import pathlib
from tqdm import tqdm
import librosa

path = pathlib.Path(r"")

valid = 0

for i in tqdm(list(path.rglob("*.wav"))):
  try:
    x , sr = librosa.load(i, sr= None)
    valid += 1
  except Exception as e:
    print(f"Error: {i}")
    continue

print(f"{valid} valid wav files")