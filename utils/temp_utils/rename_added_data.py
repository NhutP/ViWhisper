import pathlib
import argparse


def rename(start, txt_folder: pathlib.Path, mp3_folder: pathlib.Path):
    for i in mp3_folder.rglob("*.mp3"):
        new_name = str(int(i.stem) + start) + ".mp3"
        i.rename(new_name)

    for i in txt_folder.rglob("*.txt"):
        new_name = str(int(i.stem) + start) + ".txt"
        i.rename(new_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("mp3", help="path to mp3", type=str)
    parser.add_argument("txt", help="path to txt", type=str)
    parser.add_argument("start", help="start to name", type=int)

    args = parser.parse_args()

    mp3_path = pathlib.Path(args.mp3)
    txt_path = pathlib.Path(args.txt)
    start = int(args.start)

    rename(start, mp3_path, txt_path)
