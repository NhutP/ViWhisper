import pathlib
from tqdm import tqdm


au = pathlib.Path(r"/mnt/mmlab2024/datasets/ytb_craw/ytb_audio")
sc = pathlib.Path(r"")

print("Have script but not audio:")
for i in tqdm(list(sc.rglob("*.txt"))):
  co = pathlib.Path(str(i).replace(str(sc), str(au))).parent / (i.stem + '.wav')
  if not co.exists():
    print(i)

print("Have audio but not script:")
for i in tqdm(list(au.rglob("*.wav"))):
  co = pathlib.Path(str(i).replace(str(au), str(sc))).parent / (i.stem + '.txt')
  if not co.exists():
    print(i)