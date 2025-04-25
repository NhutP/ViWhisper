import sys
sys.path.insert(0, r'..')
from datasets import load_from_disk
import argparse
from whisper_train.load_data import whisper_data_loader

def check_dataset(path, split=None):
    # if split:
    #     dataset = load_from_disk(path)[split]
    # else:
    #     dataset = load_from_disk(path)
    loader = whisper_data_loader(path)
    dataset = loader.load_all()

    # print("Samples")
    # print(dataset[0])
    
    dataset = dataset.with_format('numpy')
    # print(dataset[0]['input_features'])
    # print(dataset[0]['input_features'].shape)

    print("Info")
    print(dataset)

    # print(dataset[0]['array'])
    # print(dataset[0]['array'].shape)
   
    # print(type(dataset[0]["array"]))
    # print("cus")
    # print(len(dataset["path"]))
    # print(len(list(dataset["path"])))
    # print("------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("pa", help="path to dataset", type=str)
    parser.add_argument("--split", type=str)

    args = parser.parse_args()

    check_dataset(args.pa, args.split)
