import pathlib
import shutil
from tqdm import tqdm


source = pathlib.Path(r"")
des = pathlib.Path(r"")

name_index = 1

for i in tqdm(list(source.rglob("*.wav"))):
  shutil.copy2(i, des / (i.stem) + str(name_index) + '.wav') 

