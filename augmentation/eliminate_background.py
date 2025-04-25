from audiomentations.core.utils import calculate_rms
import pathlib
import librosa
from tqdm import tqdm

path = pathlib.Path(r"/mnt/mmlab2024/datasets/VNSTT/back_ground_final/use_bgn")

deleted = 0

for i in tqdm(list(path.rglob("*.wav"))):
  x, sr = librosa.load(i, sr=None)
  if calculate_rms(x) < 1e-9:
    deleted+=1
    # i.unlink()
    print("Found")

print(f"{deleted} files were deleted.")