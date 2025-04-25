import pathlib
from tqdm import tqdm

source = r''
des = r''
source_path = pathlib.Path(source)
des_path = pathlib.Path(des)


for i in tqdm(list(source_path.rglob("*.wav"))):
  txt = pathlib.Path(str(i).replace(source, des))
  if not txt.exists():
    print(i)