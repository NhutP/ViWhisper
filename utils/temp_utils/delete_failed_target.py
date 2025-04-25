import pathlib

path = r""
path = pathlib.Path(path)

for i in path.rglob("*.txt"):
    with open(i, "r", encoding="utf8") as f:
        text = f.read().split()
        if len(text) == 0:
            print(f"Removed {i}")
            i.unlink()
