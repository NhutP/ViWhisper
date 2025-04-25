from mutagen.mp3 import MP3
import pathlib
import math
from audio_load_and_split import *
import torch
from transformers.utils.hub import cached_file
from importlib.machinery import SourceFileLoader
from IPython.lib.display import Audio
import time
import torch
from transcripter import *
import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM



class FileTranscripter(LocalSplitter):
    def __init__(self, model_inferencer, processor_feature_extractor, processor_tokenizer_decoder, processor_tensor_decoder, tensor_getter, device, sampling_rate=16000, chunk_size=5,duration=120,max_read_size=18000):
        super().__init__(sampling_rate, chunk_size, duration, max_read_size)
        # device
        self.device = device
        # model to use
        self.model_inferencer = model_inferencer
        self.processor_tokenizer_decoder = processor_tokenizer_decoder
        self.processor_feature_extractor = processor_feature_extractor
        self.processor_tensor_decoder = processor_tensor_decoder
        self.tensor_getter = tensor_getter

    def inference(self, audio_chunks):
        '''
        Inference and the code to transcript
        '''

        input_chunks = self.processor_feature_extractor(audio_chunks)
        input_chunks = input_chunks.to(self.device)

        with torch.inference_mode():
            output_chunks = self.model_inferencer(input_chunks)

        result = self.processor_tokenizer_decoder(output_chunks)

        return result
    

    def inference_to_tensor(self, audio_chunks):
        '''
        Inference but not decode yet
        '''

        input_chunks = self.processor_feature_extractor(audio_chunks)
        input_chunks.to(self.device)
        output_chunks = self.model_inferencer(input_chunks)

        return output_chunks
    

    def decode_tensor(self, inferenced_tensor):
        return self.processor_tensor_decoder(inferenced_tensor)


    def transcript_segment_from_raw_audio(self, file_path: pathlib.Path, segment: int):
        '''
        Transcript a a segment of audio (the first segment is 1)
        '''
        try:
            audio_chunks, final_audio_chunk = self.load(file_path= file_path, segment= segment)
        except Exception as exc:
            print(f'{file_path}, segment {segment} ', end='')
            print(str(exc))
            return None

        result = []

        # transcript splitted chunks in the segment
        if audio_chunks is not None:
            result = self.inference(audio_chunks)
            
        # transcript the final chunk
        if final_audio_chunk is not None:            
            result.append(self.inference(final_audio_chunk)[0])

        return result


    def transcript_raw_audio(self, file_path: pathlib.Path, starting_segment: int):
        '''
        Transcript the audio strat from a specific segment by transcripting the segments one by one
        '''
        # duration of the audio file
        file_duration = MP3(file_path).info.length
        # number of segment that the audio file has
        num_of_segment = int(file_duration / self.duration) + int((file_duration - self.duration * int(file_duration / self.duration)) > 0)

        result = [0] * num_of_segment

        # transcript segment one by one
        for segment in range(starting_segment, num_of_segment + 1, 1):
            result[segment - 1] = self.transcript_segment_from_raw_audio(file_path, segment)

        return result


    def transcript_write_raw_audio(self, raw_audio: pathlib.Path, raw_transcript_file_path: pathlib.Path, chunked_transcript_folder: pathlib.Path, starting_segment: int):
        '''
        Transcript the audio strat from a specific segment by transcripting the segments one by one and write to a text file.
        Also, save the reading state to a json file.
        '''

        # duration of the audio file
        file_duration = MP3(raw_audio).info.length
        # number of segment that the audio file has
        num_of_segment = int(file_duration / self.duration) + int((file_duration - self.duration * int(file_duration / self.duration)) > 0)

        current_chunk = 1
    
        for segment in range(starting_segment, num_of_segment + 1, 1):
            # transcript the segment
            segment_transcript = self.transcript_segment_from_raw_audio(raw_audio, segment)

            transcript  = ''
            for chunk_transcript in segment_transcript:
                transcript += (chunk_transcript + '\n')
                # write to the segment txt
                with open(chunked_transcript_folder / (self.get_name(current_chunk) + '.txt'), 'w', encoding='utf8') as transcript_write:
                    transcript_write.write(chunk_transcript)
                current_chunk += 1

            # write to the raw transcript file
            with open(raw_transcript_file_path, 'a', encoding='utf8') as transcript_write:
                transcript_write.write(transcript)
    
    
    def transcript_chunked_audio_segment(self, chunked_audio_folder: pathlib.Path, segment_index):
        '''
        Inference and decode a segment in the audio by its chunked 
        '''

        audio_chunks = self.load_chunked_segment(chunked_audio_folder, segment_index)
        result = self.inference(audio_chunks)
        return result


    def transcript_chunked_audio_segment_to_tensor(self, chunked_audio_folder: pathlib.Path, segment_index):
        audio_chunks = self.load_chunked_segment(chunked_audio_folder, segment_index)
        result_tensors = self.inference_to_tensor(audio_chunks)
        return result_tensors
    

    def transcript_write_file_chunked_audio(self, chunked_audio_folder: pathlib.Path, raw_transcript_file: pathlib.Path, chunked_transcript_folder: pathlib.Path):
        num_of_chunk_per_segment = int(self.duration / self.chunk_size) 
        num_of_chunk = len(list(chunked_audio_folder.iterdir()))
        num_of_segment = math.ceil(num_of_chunk / num_of_chunk_per_segment)

        current_chunk = 1

        # transcript segment by segment
        for segment in range(1, num_of_segment + 1):
            transcript = ''
            audio_chunks = self.load_chunked_segment(chunked_audio_folder, num_of_chunk, segment)
            segment_transcript = self.inference(audio_chunks)

            # write to raw_transcript file and chunked_transcript file
            for i in range(len(segment_transcript)):
                with open(chunked_transcript_folder / (chunked_transcript_folder.name + '_' + self.get_name(current_chunk) + '.txt'), 'w', encoding='utf8') as chunk_writer:
                    chunk_writer.write(segment_transcript[i])
                transcript += (segment_transcript[i] + '\n')
                current_chunk += 1

            with open(raw_transcript_file, 'a', encoding='utf8') as segment_writer:
                segment_writer.write(transcript)


    def transcript_file_to_chunked_tensor(self, chunked_audio_folder: pathlib.Path, chunked_transcripted_tensor_folder: pathlib.Path):
        num_of_chunk_per_segment = int(self.duration / self.chunk_size) 
        num_of_chunk = len(list(chunked_audio_folder.iterdir()))
        num_of_segment = math.ceil(num_of_chunk / num_of_chunk_per_segment)

        current_chunk = 1

        for id in range(1, num_of_segment + 1):
            audio_chunks = self.load_chunked_segment(chunked_audio_folder, num_of_chunk, id)
            segment_transcript_tensort_output = self.inference_to_tensor(audio_chunks)

            segment_transcript_tensor = self.tensor_getter(segment_transcript_tensort_output)
            for i in range(segment_transcript_tensor.shape[0]):
                torch.save(segment_transcript_tensor[i], chunked_transcripted_tensor_folder / (chunked_transcripted_tensor_folder.name + '_' + self.get_name(current_chunk) + '.pt'))
                current_chunk += 1


class FolderTranscripter(FileTranscripter):
    def __init__(self, raw_audio_folder, chunked_audio_folder, raw_transcript_folder, chunked_transcript_folder, chunked_transcripted_tensor_folder, model_inferencer, processor_feature_extractor, processor_tokenizer_decoder, processor_tensor_decoder, tensor_getter ,device, sampling_rate=16000, chunk_size=30, duration=60, max_read_size=18000):
      
        super().__init__(model_inferencer, processor_feature_extractor, processor_tokenizer_decoder, processor_tensor_decoder, tensor_getter, device, sampling_rate, chunk_size, duration, max_read_size)

        # create necessary folders if not exist
        self.raw_audio_folder = pathlib.Path(raw_audio_folder)
        self.chunked_audio_folder = pathlib.Path(chunked_audio_folder)
        
        self.raw_transcript_folder = pathlib.Path(raw_transcript_folder)
        self.chunked_transcript_folder = pathlib.Path(chunked_transcript_folder)
        
        # check chunked tensor folder
        if chunked_transcripted_tensor_folder is not None:
            self.chunked_transcripted_tensor_folder = pathlib.Path(chunked_transcripted_tensor_folder)
            if not self.chunked_transcripted_tensor_folder.exists():
                self.chunked_transcripted_tensor_folder.mkdir()
                print(f'Created {self.chunked_transcripted_tensor_folder}')

        assert self.raw_audio_folder.exists() == True

        if not self.chunked_audio_folder .exists():
            self.chunked_audio_folder.mkdir()
            print(f'Created {self.chunked_audio_folder}')

        if not self.raw_transcript_folder.exists():
            self.raw_transcript_folder.mkdir()
            print(f'Created {self.raw_transcript_folder}')
        
        if not self.chunked_transcript_folder.exists():
            self.chunked_transcript_folder.mkdir()
            print(f'Created {self.chunked_transcript_folder}')
        
        

    def transcript_write_folder_raw_audio(self):
        '''
        Transcript all audio files in the folder by transcripting its files one by ones.
        Also, save the reading state to a json file.
        '''
        finished_files = []

        # create a baseline-transcript folder to store transcripted text file if it not exists
        if not self.raw_transcript_folder.exists():
            self.raw_transcript_folder.mkdir(parents=False)
            print('Create baseline-transcript folder')
        else:

        # if the transcript folder, save all the file names
            finished_files = [str(file.stem) for file in self.raw_transcript_folder.iterdir()]

        # transcript audios in the folder
        for audio_file_path in self.raw_audio_folder.iterdir():
            audio_file_name = str(audio_file_path.stem)

            # skip the audio files that has already been completed
            if  audio_file_name in finished_files:
                print(f'Already complete {self.raw_audio_folder / audio_file_name}\n--------------------------------------')
                continue

            # create text file to store transcript (same name as its audio)
            transcript_file_name = str(audio_file_path.stem) + '.txt'
            self.raw_transcript_folder.touch(transcript_file_name)

            # transcript the file
            print(f'Transcripting {audio_file_path}')
            self.transcript_write_raw_audio(audio_file_path, self.raw_transcript_folder / transcript_file_name, self.chunked_transcript_folder)

            finished_files.append(audio_file_name)
            print(f'Completed {self.data_folder / audio_file_name}\n---------------------------------------')

        print(f'COMPLETED ALL AUDIO IN FOLDER {self.raw_audio_folder}')
    

    def transcript_write_chunked_audio_folder(self):
        # list of transcripted file
        finished_file = [file.name for file in self.chunked_transcript_folder.iterdir()]
        
        for chunked_audio_folder in self.chunked_audio_folder.iterdir():
            audio_file_name = chunked_audio_folder.name

            # skip the completed file
            if audio_file_name in finished_file:
                print(f"Already complpeted {chunked_audio_folder}")
                continue

            # create text file to store transcript (same name as its audio)
            raw_transcript_file_name = str(audio_file_name) + '.txt'
            self.raw_transcript_folder.touch(raw_transcript_file_name)

            raw_transcript_file = self.raw_transcript_folder / raw_transcript_file_name
            temp_chunked_transcript_folder = self.chunked_transcript_folder / audio_file_name 

            if not temp_chunked_transcript_folder.exists():
                temp_chunked_transcript_folder.mkdir()

            # transcript the file
            print(f'Transcripting {chunked_audio_folder}')

            with torch.inference_mode():
                self.transcript_write_file_chunked_audio(chunked_audio_folder, raw_transcript_file, temp_chunked_transcript_folder)

            print(f'Transcripted {chunked_audio_folder}\n---------------------------------------')

        print(f'TRANSCRIPTED ALL AUDIOS IN FOLDER {self.raw_audio_folder}')

    
    def transcript_folder_to_tensor(self):
        finished_file = [file.name for file in self.chunked_transcripted_tensor_folder.iterdir()]

        for chunked_audio_folder in self.chunked_audio_folder.iterdir():
            audio_file_name = chunked_audio_folder.name

            # skip the completed file
            if audio_file_name in finished_file:
                continue

            temp_chunked_transcripted_tensor_folder = self.chunked_transcripted_tensor_folder / audio_file_name

            if not temp_chunked_transcripted_tensor_folder.exists():
                temp_chunked_transcripted_tensor_folder.mkdir()
            
            print(f'Transcripting to tensor {chunked_audio_folder}')

            with torch.inference_mode():
                self.transcript_file_to_chunked_tensor(chunked_audio_folder, temp_chunked_transcripted_tensor_folder)

            print(f'Transcripted {chunked_audio_folder}\n---------------------------------------')

        print(f'TRANSCRIPTED TO TENSOR ALL AUDIOS IN FOLDER {self.raw_audio_folder}')

    
    def decode_write_chunked_tensor_folder(self):
        '''
        Decode the tensor to transcript
        '''

        finished_files = [file.name for file in self.chunked_transcript_folder.iterdir()]

        with torch.inference_mode():
            for temp_chunked_tensor_folder in self.chunked_transcripted_tensor_folder.iterdir():
                audio_name = temp_chunked_tensor_folder.name

                if audio_name in finished_files:
                    continue
                
                temp_chunked_transcript_folder = self.chunked_transcript_folder / audio_name
                temp_chunked_transcript_folder.mkdir()

                print(f'Decoding tensors in {temp_chunked_tensor_folder}')
                for tensor_file in (temp_chunked_tensor_folder).iterdir():
                    with open(tensor_file, 'rb') as tensor_loader:
                        inferenced_tensor = torch.load(tensor_loader)
                        temp_chunked_transcript = self.decode_tensor(inferenced_tensor)
                    
                    with open(temp_chunked_transcript_folder / (tensor_file.stem + '.txt') , 'w', encoding='utf8') as trans_writer:
                        trans_writer.write(temp_chunked_transcript)

                    with open(self.raw_transcript_folder / (audio_name + '.txt'), 'a', encoding='utf8') as raw_writer:
                        raw_writer.write(temp_chunked_transcript + '\n')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('ra', help='path to raw audio folder', type=str)
    parser.add_argument('ca', help='path to chunked audio folder', type=str)
    parser.add_argument('rt', help='path to raw transcript folder', type=str)
    parser.add_argument('ct', help='path to chunked transscript folder', type=str)
    
    parser.add_argument('v', help = 'version of the model')
    
    parser.add_argument('--tf', help= 'chunked tensor folder', type=str)
    parser.add_argument('--task', help= 'task to do (1, 2, 3) or 0 (transcript without create tensor)', type=str, default='0')

    parser.add_argument('--lm', help = 'to use language model', choices=['y', 'n'], default='n')
    parser.add_argument('--cs', help = 'chunk size', default=30, type=int)
    parser.add_argument('-sr', '-sampling_rate', help = 'sampling rate', default=16000, type=int)
    parser.add_argument('--du' , help = 'duration to read each time', default=120, type=int)
    parser.add_argument('--mr', help = 'max size to read when split audio', default=18000, type=int)

    args = parser.parse_args()

    raw_audio_folder = args.ra
    chunked_audio_folder = args.ca
    raw_transcript_folder = args.rt
    chunked_transcript_folder = args.ct
    chunked_tensor_folder = args.tf

    task_instructor = args.task

    model_version = args.v
    is_use_lm = args.lm
    duration = args.du
    sampling_rate = args.sr 
    max_read_size = args.mr
    chunk_size = args.cs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   
    # load model without language model
    if model_version == 'base':
        model_name = "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"
    elif model_version == 'large':
        model_name = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
    
    if is_use_lm == 'n':
        model = SourceFileLoader("model", cached_file(model_name,filename="model_handling.py")).load_module().Wav2Vec2ForCTC.from_pretrained(model_name)
        processor = Wav2Vec2Processor.from_pretrained(model_name)

        model.to(device)

        model_inferencer = lambda input : model(**input)
        processor_feature_extractor = lambda chunks : processor.feature_extractor(chunks, sampling_rate=sampling_rate, return_tensors='pt')
        processor_tokenizer_decoder = lambda chunks :[processor.tokenizer.decode(chunks.logits[i].unsqueeze(0).argmax(dim=-1)[0].detach().cpu().numpy()) for i in range(chunks.logits.shape[0])]
        processor_tensor_decoder = lambda tensor :  processor.tokenizer.decode(tensor.unsqueeze(0).argmax(dim=-1)[0].detach().cpu().numpy())

    # load model with language model
    elif is_use_lm == 'y':
        model = SourceFileLoader("model", cached_file(model_name,filename="model_handling.py")).load_module().Wav2Vec2ForCTC.from_pretrained(model_name)
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)

        model.to(device)

        model_inferencer = lambda input : model(**input)
        processor_feature_extractor = lambda chunks : processor.feature_extractor(chunks, sampling_rate=sampling_rate, return_tensors='pt')
        processor_tokenizer_decoder = lambda chunks : [processor.decode(chunks.logits[i].unsqueeze(0).cpu().detach().numpy()[0], beam_width=100).text for i in range(chunks.logits.shape[0])]
        processor_tensor_decoder = lambda tensor : processor.decode(tensor.unsqueeze(0).cpu().detach().numpy()[0], beam_width=100).text 
            
        tensor_getter = lambda output : output.logits


    folder_transcript = FolderTranscripter(raw_audio_folder, chunked_audio_folder, raw_transcript_folder, chunked_transcript_folder, chunked_tensor_folder, model_inferencer, processor_feature_extractor, processor_tokenizer_decoder, processor_tensor_decoder, tensor_getter, device, duration=duration, sampling_rate=sampling_rate, chunk_size=chunk_size, max_read_size=max_read_size)

    print('Device:')
    print(device)

    start = time.time()

    if '0' in task_instructor:
        folder_transcript.split_save_folder(folder_transcript.raw_audio_folder, folder_transcript.chunked_audio_folder)
    if '1' in task_instructor:
        folder_transcript.transcript_write_chunked_audio_folder()
    if '2' in task_instructor:
        folder_transcript.transcript_folder_to_tensor()
    if '3' in task_instructor:
        folder_transcript.decode_write_chunked_tensor_folder()

    end = time.time()

    print(end-start)