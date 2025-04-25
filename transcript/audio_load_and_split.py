import sys
sys.path.insert(0, r'..')

import librosa
import numpy as np
from mutagen.mp3 import MP3
from typing import Tuple
from pydub import *
import pathlib
import soundfile as sf
import math
import argparse
from tqdm import tqdm
from utils.memory import avoid_OOM


class LocalLoader():
    def __init__(self, sampling_rate=16000, duration=120):
        self.sampling_rate = sampling_rate

        # decide duration for each time read an audio
        self.duration = duration


    def load(self, file_path: pathlib.Path, segment: int):
        '''
        Load audio with a specific segment.
        '''
        audio = None
        file_sampling_rate = None
        startingTime = self.duration*(segment - 1)

        if MP3(file_path).info.length >= startingTime:
            # if segment satisfied, read the segment
            audio, file_sampling_rate = librosa.load(file_path, sr=self.sampling_rate, offset = startingTime, duration=self.duration)
        else:
            # raise out of segment
            raise Exception('Out of segment at' + str(file_path))
        
        return audio, file_sampling_rate


    def audio_by_silent(self, audio, top_db, frame_length, max_duration, min_duration, depth=0):
        if audio.shape[0] < self.sampling_rate * min_duration:
            return [[], audio]
        
        avoid_OOM(60)

        hop_length = max(512, int(frame_length / 8))
        silent_timestamp = librosa.effects.split(y=audio, frame_length=frame_length, top_db=top_db, hop_length=hop_length)

        accumulate_audio = None
        result_splited_audios = []

        for i in range(silent_timestamp.shape[0]):
            avoid_OOM(60)
            splited_audio = audio[silent_timestamp[i][0] : silent_timestamp[i][1]]

            if accumulate_audio is not None:
                splited_audio = np.concatenate([accumulate_audio, splited_audio], axis=0)
                accumulate_audio = None

            if splited_audio.shape[0] < self.sampling_rate * min_duration:
                accumulate_audio = splited_audio
                continue

            elif splited_audio.shape[0] > max_duration * self.sampling_rate:
                recur_result = self.audio_by_silent(splited_audio, int(top_db), int(frame_length / 2), max_duration, min_duration, depth+1)
                result_splited_audios += recur_result[0]
                if recur_result[1] is not None:
                    accumulate_audio = recur_result[1]
                continue

            result_splited_audios.append(splited_audio)

        return [result_splited_audios, accumulate_audio]



class LocalSplitter(LocalLoader):
    def __init__(self, sampling_rate=16000, chunk_size=5, duration=120, max_read_size=18000):
        assert (duration % chunk_size) == 0
        assert (max_read_size % chunk_size) == 0
        super().__init__(sampling_rate, duration)
        self.chunk_size = chunk_size
        self.get_name = lambda index : f"{(index):05n}"
        self.max_read_size = max_read_size


    def load_chunked_segment(self, chunked_audio_folder: pathlib.Path, total_chunk, segment_index = 1):
        chunk_per_segment =int(self.duration / self.chunk_size)
        start_chunk = (segment_index - 1)*chunk_per_segment + 1
        end_chunk = min(segment_index * chunk_per_segment, total_chunk)

        audio = np.zeros((end_chunk - start_chunk + 1, self.chunk_size * self.sampling_rate), dtype=float)

        for i in range(start_chunk, end_chunk + 1, 1):
            audio[i - start_chunk] = librosa.load(chunked_audio_folder / (chunked_audio_folder.name + '_' + self.get_name(i) + '.wav'), sr=self.sampling_rate)[0]
        
        return audio
        

    def load(self, file_path: pathlib.Path, segment: int):
        '''
        Load and split the audio to some chunks (with a given size).
        The audio file is loaded with a specific segment and duration. 
        '''
        #load audio
        fullAudio, _ = super().load(file_path,segment)
        #number of floats need to use to store a chunk of audio
        number_per_chunk = self.chunk_size * self.sampling_rate
        # the size of final chunk (the final chunk may be shorter than the others)
        finalChunk_size = fullAudio.shape[0] % number_per_chunk
        
        # read chunks from audio except for the final chunk
        audio_chunks = None
        if int(fullAudio.shape[0] / number_per_chunk) > 0:
            audio_chunks = fullAudio[0 : fullAudio.shape[0] - finalChunk_size].reshape(int(fullAudio.shape[0] / number_per_chunk), -1)

        # final chunk in the audio
        final_chunk = None
        # read the final chunk
        if finalChunk_size > 0:
            final_chunk = np.zeros(number_per_chunk)
            final_chunk[0 : finalChunk_size] = fullAudio[fullAudio.shape[0] - finalChunk_size : fullAudio.shape[0]]
        
        return audio_chunks, final_chunk


    def split_and_save_fixed(self, audio_file: pathlib.Path, result_folder: pathlib.Path):
        # duration of the audio file
        file_duration = MP3(audio_file).info.length

        input_fileName_withoutExtension = audio_file.stem
        read_time = math.ceil(file_duration / self.max_read_size)

        for read_time_i in tqdm(range(read_time)):
            audio, _ = librosa.load(audio_file, sr=self.sampling_rate, offset=read_time_i * self.max_read_size, duration=self.max_read_size)
            # number of segment that the audio file has
            num_of_complete_chunk = int(audio.shape[0] / (self.chunk_size * self.sampling_rate))

            start_index = read_time_i * int(self.max_read_size / self.chunk_size)

            for index in range(num_of_complete_chunk):
                start = index * self.chunk_size * self.sampling_rate
                end = (index + 1) * self.chunk_size * self.sampling_rate
                split_audio = audio[start:end]
                output_file_name = result_folder / (input_fileName_withoutExtension + '_' + f"{(start_index+index+1):09n}.wav")
                output_path = pathlib.Path(output_file_name)
                sf.write(output_path, split_audio, self.sampling_rate, format='mp3')

            if audio.shape[0] % (self.chunk_size * self.sampling_rate) != 0:
                start = (num_of_complete_chunk) * self.chunk_size * self.sampling_rate
                end = audio.shape[0]
                silent = np.zeros((self.chunk_size*self.sampling_rate - audio.shape[0] % (self.chunk_size * self.sampling_rate),), dtype=float)

                final_chunk = np.concatenate((audio[start : end], silent))
                output_file_name = result_folder / (input_fileName_withoutExtension + '_' + f"{(start_index + num_of_complete_chunk+1):09n}.wav")
                output_path = pathlib.Path(output_file_name)
                sf.write(output_path, final_chunk, self.sampling_rate, format='mp3')


    def split_save_folder_fixed(self, raw_audio_folder: pathlib.Path, result_folder: pathlib.Path):
        raw_files = list(raw_audio_folder.rglob('*.wav'))

        for i in range(len(raw_files)):
            result_dir = pathlib.Path(str(raw_files[i].parent).replace(str(raw_audio_folder), str(result_folder))) / raw_files[i].stem
            if not result_dir.exists():
                result_dir.mkdir(parents=True)
                print(f'Created {result_dir}')
            else:
                print(f'{result_dir} exists\n-------------------------------------------')
                continue
            print(f"Spliting {raw_files[i]} and store at {result_dir}")
            self.split_and_save_fixed(raw_files[i], result_dir)

    
    def split_save_by_silent(self, audio_file: pathlib.Path, result_folder: pathlib.Path, top_db, frame_length, max_duration, min_duration):
        name_index = 1
        read_time = 0     

        print(f"Loading {audio_file}")
        full_audio, _ = librosa.load(audio_file, sr=self.sampling_rate)
        total_duration = full_audio.shape[0] / self.sampling_rate
        total_read_time = math.ceil(total_duration / self.max_read_size)

        for i in tqdm(range(total_read_time)):
            start_read = self.sampling_rate*self.max_read_size*read_time
            end_read = min(full_audio.shape[0], self.sampling_rate*self.max_read_size*(read_time + 1))
            audio = full_audio[start_read : end_read]
            avoid_OOM()

            if audio.shape[0] < self.sampling_rate * min_duration:
                break

            result_audios, _ = self.audio_by_silent(audio, top_db, frame_length, max_duration, min_duration)
            print(f"There are {len(result_audios)} files")
            for i in range(len(result_audios)):
                sf.write(result_folder / (audio_file.stem + f'_{(name_index):09n}' + '.wav'), result_audios[i], self.sampling_rate)
                name_index += 1

            read_time += 1

        print(f"Total: {name_index - 1} files")


    def split_save_silent_folder(self, raw_audio_folder: pathlib.Path, result_folder: pathlib.Path, top_db, frame_length=32000, max_duration=12, min_duration=3):
        
        raw_files = list(raw_audio_folder.rglob('*.wav'))

        for i in range(len(raw_files)):
            avoid_OOM(60)
            result_dir = pathlib.Path(str(raw_files[i].parent).replace(str(raw_audio_folder), str(result_folder))) / raw_files[i].stem
            if not result_dir.exists():
                result_dir.mkdir(parents=True)
                print(f'Created {result_dir}')
            else:
                print(f'{result_dir} exists\n-------------------------------------------')
                continue
            print(f"Spliting {raw_files[i]} by silence and store at {result_dir}")
            self.split_save_by_silent(raw_files[i], result_dir, top_db, frame_length, max_duration, min_duration)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('ra', help='path to raw audio folder', type=str)
    parser.add_argument('out', help='path to output folder', type=str)
    parser.add_argument('task', help='task to do (0 if split by fixed size, 1 if load by silent)', type=str)

    parser.add_argument('--topdb', help='threshold to consider silent in db', type=int, default=30)
    parser.add_argument('--mad', help='max duration for each file', type=int, default=12)
    parser.add_argument('--mid', help='min duration for each file', type=int, default=3)
    parser.add_argument('--framlen', help='frame length to consider silent (if all numbers in a frame length smaller than a threshold, it is silence)', type=int, default=32000)
    parser.add_argument('--mrs', help='max read size', type=int, default=18000)

    args = parser.parse_args()

    task = args.task
    raw_audio_path = pathlib.Path(args.ra)
    output_path = pathlib.Path(args.out)

    max_duration = int(args.mad)
    min_duration = int(args.mid)
    top_db = int(args.topdb)
    frame_length = int(args.framlen)
    max_read_size = int(args.mrs)

    local_split = LocalSplitter(max_read_size=max_read_size)

    if '0' in task:
        local_split.split_save_folder_fixed(raw_audio_path, output_path)
    if '1' in task:
        local_split.split_save_silent_folder(raw_audio_path, output_path, top_db, frame_length,  max_duration, min_duration)