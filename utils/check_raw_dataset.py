import sys
sys.path.insert(0, r'..')
from datasets import load_dataset, concatenate_datasets
import argparse
from prepare_data import list_leaf_dirs
import pathlib
from tqdm import tqdm

def check_dataset(path):
    path = pathlib.Path(path)
    leafs = list_leaf_dirs(path)
    
    dataset = None
    if len(leafs) == 0:
        dataset = load_dataset("audiofolder", data_dir=str(path))
    else:
        for i in tqdm(leafs):
            temp_dataset = load_dataset("audiofolder", data_dir=str(i))
            if dataset is not None:
                dataset = concatenate_datasets([dataset, temp_dataset])
            else:
                dataset = temp_dataset

    print(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("pa", help="path to dataset", type=str)

    args = parser.parse_args()

    check_dataset(args.pa)
