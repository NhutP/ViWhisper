import pathlib
import librosa
import pandas as pd
import argparse
from tqdm import tqdm
import mutagen
from mutagen.wave import WAVE


def list_leaf_dirs(root_dir: pathlib.Path):
    root_dir = pathlib.Path(root_dir)
    leaf_dirs = []
    for path in root_dir.rglob("*"):
        if path.is_dir():
            is_leaf = True
            for i in path.iterdir():
                if i.is_dir():
                    is_leaf = False
            if is_leaf:
                leaf_dirs.append(path)

    return leaf_dirs


def get_duration(path):
    audio = WAVE(str(path))

    # contains all the metadata about the wavpack file
    audio_info = audio.info
    length = audio_info.length

    return length / 3600


def sum_duration(path: pathlib.Path):
    total = 0

    for i in path.rglob("*.wav"):
        total += get_duration(i)

    return total


def get_info(raw_folder: pathlib.Path, splited_folder: pathlib.Path, csv_path):
    info = {"name": [], "raw_duration": [], "splited_duration": []}
    raw_audios = list(raw_folder.rglob("*.wav"))
    splited_audios = [
        pathlib.Path(
            str(i.parent / i.stem).replace(str(raw_folder), str(splited_folder))
        )
        for i in raw_audios
    ]

    for i in tqdm(range(len(raw_audios))):
        name = str(raw_audios[i])
        raw_du = get_duration(raw_audios[i])
        splt_du = sum_duration(splited_audios[i])

        info["name"].append(name)
        info["raw_duration"].append(raw_du)
        info["splited_duration"].append(splt_du)

        print(name)
        print(raw_du)
        print(splt_du)

    if csv_path is not None:
        pd.DataFrame.from_dict(info).to_csv(csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("ra", help="path to raw audio folder", type=str)
    parser.add_argument("spl", help="path to splited folder", type=str)
    parser.add_argument("--csv", help="csv path", type=str)

    args = parser.parse_args()

    raw_folder = pathlib.Path(args.ra)
    splited_folder = pathlib.Path(args.spl)
    csv = pathlib.Path(args.csv) if args.csv is not None else None

    get_info(raw_folder, splited_folder, csv)
