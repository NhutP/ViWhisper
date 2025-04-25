import argparse
import pathlib
import csv
#from mutagen.mp3 import MP3
import librosa
from tqdm import tqdm
import wave

def get_wav_duration(file_path):
    # Open the .wav file
    with wave.open(file_path, 'rb') as wav_file:
        # Extract the number of frames
        n_frames = wav_file.getnframes()
        # Extract the sample rate
        frame_rate = wav_file.getframerate()
        
        # Calculate the duration
        duration = n_frames / float(frame_rate)
    return duration

# def get_wav_duration(file_path):
#     x, sr = librosa.load(file_path, sr=None)
#     return x.shape[0] / sr

def count_files(dir_path :pathlib.Path):
    time = 0
    for i in tqdm(list(dir_path.rglob('*.wav'))):
        try:
            time += get_wav_duration(str(i))
        except Exception as e:
            print(f"Error: {i}")
            continue
    print(dir_path)
    print(time / 3600)
    print('---------------------------')


def count_perfect(au_path :pathlib.Path, aligned_path: pathlib.Path):
    time = 0
    for i in tqdm(list(aligned_path.rglob('*.txt'))):
        if '_1' not in i.parent.name:
            continue

        au_file = pathlib.Path(str(i).replace(str(aligned_path), str(au_path))).parent.parent / (i.stem + '.wav')
        time += get_wav_duration(str(au_file))
       
    print(time / 3600)
    print('---------------------------')

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('aufo', help='audio folder', type=str)
    parser.add_argument('--alfo', help='algined folder if count perfect', type=str)
    args = parser.parse_args()

    paths = [pathlib.Path(i) for i in str(args.aufo).split('+')]
    
    if args.alfo:
        count_perfect(pathlib.Path(args.aufo), pathlib.Path(args.alfo))
    else: 
        for i in paths:
            count_files(i)