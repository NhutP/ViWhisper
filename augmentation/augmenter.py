import sys
sys.path.insert(0, r'..')

from augmenter_module import augmenter_module
import pathlib
import pandas as pd
import argparse
import librosa
import soundfile as sf
from tqdm import tqdm
import shutil
from utils.prepare_data import list_leaf_dirs
from utils.memory import avoid_OOM

import numpy
from numpy.random import randint
import random


space = '\n------------------------------------------\n'

# class audio_augmenter(audio_augmenter_module):
class audio_augmenter(augmenter_module):
  def __init__(self, background_strogage):
    super().__init__(background_strogage)

    self.get_effect = \
    self.backgrounds_high_transformers + \
    self.backgrounds_average_transformers + \
    self.background_Gauss_compose + \
    self.background_BandPass_compose + \
    self.background_tanhdistortion_compose + \
    self.background_PitchShift_compose + \
    [self.Gauss_SNR_high_transformer] * 1000 + \
    [self.Gauss_SNR_average_transformer] * 1000 +\
    [self.BandPass_transformer] * 5000 + \
    [self.ClippingDistortion_transformer] * 5000 + \
    [self.TanhDistortion_transformer] * 5000 +\
    [self.PitchShift_transformer] * 5000
    
    random.shuffle(self.get_effect)
    

  def prepare_dir(self, original_root_dir: pathlib.Path, output_root_dir):
    for i in original_root_dir.rglob('*'):
      if i.is_dir():
        correspod_dir = str(i).replace(str(original_root_dir), str(output_root_dir))

        if not pathlib.Path(correspod_dir).exists():
          pathlib.Path(correspod_dir).mkdir(parents=True)


  def generate_augment_info(self, original_root_dir: pathlib.Path, output_root_dir, csv_store=None):
    audio_files = list(original_root_dir.rglob('*.wav'))
    audio_files = list(map(str, audio_files))

    random_effect = [((i % len(self.get_effect)) + 1) for i in range(len(audio_files))]

    output_files = [i.replace(str(original_root_dir), str(output_root_dir)) for i in audio_files]

    augment_info_dict = {"audio" : audio_files, 'effect_index' : random_effect, 'out_files' : output_files}

    # if specify csv_path, store at csv file
    if csv_store is not None:
      pd.DataFrame.from_dict(augment_info_dict).to_csv(csv_store, index=False)
      print(f'Export augment info csv  at {csv_store}')
      print(space)

    return augment_info_dict
  

  def generate_augment_info_random_effect(self, original_root_dir: pathlib.Path, output_root_dir, csv_store=None):
    audio_files = list(original_root_dir.rglob('*.wav'))
    audio_files = list(map(str, audio_files))
    
    random.shuffle(audio_files)

    random_effect = [randint(0, len(self.get_effect)) for _ in range(len(audio_files))]

    output_files = [i.replace(str(original_root_dir), str(output_root_dir)) for i in audio_files]

    augment_info_dict = {"audio" : audio_files, 'effect_index' : random_effect, 'out_files' : output_files}

    # if specify csv_path, store at csv file
    if csv_store is not None:
      pd.DataFrame.from_dict(augment_info_dict).to_csv(csv_store, index=False)
      print(f'Export augment info csv  at {csv_store}')
      print(space)

    return augment_info_dict
  

  def augment_data(self, original_root_dir, output_root_dir, csv_path=None, gen_new_infor = True, augment_per_num_files = 1):
    if gen_new_infor:
      avoid_OOM(50)
      print("Prepare dir")
      audio_augment.prepare_dir(original_root_dir, output_root_dir)

      avoid_OOM(50)
      print("Get augment infor")
      augment_info = self.generate_augment_info(original_root_dir, output_root_dir, csv_path)
      audio_files = augment_info['audio']
      random_effect = augment_info['effect_index'] 
      output_files = augment_info['out_files']

    else:
      print(f"Get info from {csv_path}")
      df = pd.read_csv(csv_path)
      audio_files = df['audio'].tolist()
      random_effect = df['effect_index'].tolist()
      output_files = df['out_files'].tolist()


    print("Start augment")
    for i in tqdm(range(len(audio_files))):
      avoid_OOM(80)
      effect_index = random_effect[i]
      input_file = audio_files[i]
      output_path = output_files[i]
      
      if pathlib.Path(output_path).exists():
        # print("skip")
        continue
      # print("do")
      original_audio, sr = librosa.load(input_file, sr=16000)
      if i % augment_per_num_files == 0:
        augmented_audio = self.get_effect[effect_index - 1](original_audio, sr)
        sf.write(output_path, augmented_audio, sr)
      else:
        sf.write(output_path, original_audio, sr)

    print("Copying metadata")
    csv_dir = list_leaf_dirs(original_root_dir)

    for i in tqdm(csv_dir):
      des_csv =  pathlib.Path(str(i).replace(str(original_root_dir), str(output_root_dir))) / 'metadata.csv'
      shutil.copy2(i / 'metadata.csv', des_csv)
  

  def augment_ramdom_single(self, audio_array, sampling_rate=16000):
    if randint(0, 2) == 1:
      print("Augmented")
      return self.get_effect[randint(0, len(self.get_effect))](audio_array, sampling_rate)
    else:
      return audio_array
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('inor', help='root original audio folder', type=str)
  parser.add_argument('out', help='root output audio folder', type=str)
  parser.add_argument('--bgstr', help= 'background strogage', type=str)
  parser.add_argument('--csvp', help= 'csv store augmentation info, leave blank if not need', type=str)
  parser.add_argument('--geninfo', help= 'generate new effect or not', type=str, default='yes')
  parser.add_argument('--augpnf', help= 'number of file per 1 augmentation file', type=int, default=1)
  parser.add_argument('--seed', help= 'seed', type=int, default=1061)
  
  args = parser.parse_args()

  original_dir = pathlib.Path(args.inor)
  out_dir = pathlib.Path(args.out)
  background_strogage = pathlib.Path(args.bgstr) if args.bgstr is not None else None
  csv_path = pathlib.Path(args.csvp) if args.csvp is not None else None
  gen_in = bool(args.geninfo == 'yes')
  augment_per = int(args.augpnf)
  seed = int(args.seed)

  random.seed(seed)
  numpy.random.seed(seed)

  audio_augment = audio_augmenter(background_strogage=background_strogage)
  print(f"There are {audio_augment.num_of_background} background noises in total")
  audio_augment.augment_data(original_dir, out_dir, csv_path, gen_in, augment_per)