import sys
sys.path.insert(0, r'../..')

from utils.prepare_data import format_string
import argparse
from datasets import load_dataset, Audio
import soundfile as sf
import pathlib
import pandas as pd
from tqdm import tqdm

def down_to_wav(dataset_path, des_path):
    dict = {'file_name': [], 'transcription': []}
    des_path = pathlib.Path(des_path)
    dataset = load_dataset(dataset_path, split="test")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print('Saving')
    for i in tqdm(range(len(dataset))):
        name = f"{i + 1}.wav"
        array = dataset[i]['audio']['array']
        sf.write(des_path / name, array, 16000, format='wav')
        dict['file_name'].append(name)
        dict['transcription'].append(format_string(dataset[i]['sentence'].lower()))
    
    pd.DataFrame.from_dict(dict).to_csv(des_path / 'metadata.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help="Path to the Hugging Face dataset.")
    parser.add_argument('des_path', type=str, help="Destination path to save the wav files and metadata.")
    
    args = parser.parse_args()
    
    down_to_wav(args.dataset_path, args.des_path)

if __name__ == "__main__":
    main()