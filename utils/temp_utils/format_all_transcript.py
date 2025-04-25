import pathlib
import tqdm

import sys
sys.path.insert(0, r'../..')
from utils.prepare_data import format_string 

path = pathlib.Path(r"C:\Users\quang\Desktop\testalign2\query")


for i in tqdm.tqdm(list(path.rglob("*.txt"))):
    with open(i, "r", encoding="utf8") as r:
        text = format_string(r.read())

    with open(i, "w", encoding="utf8") as w:
        w.write(text)
