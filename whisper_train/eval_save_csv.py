import sys
sys.path.insert(0, r'..')
import pathlib

from datasets import load_from_disk, load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperForConditionalGeneration

import argparse
import torch
import pandas as pd
from utils.prepare_data import prepare_map_for_eval_transcipt
from utils.memory import avoid_OOM
from tqdm import tqdm
import evaluate
assert torch.cuda.is_available()

wer = evaluate.load("wer")

def eval_save_csv(pipe, data_dir, save_csv, batch_size = 32):
  data_dir = pathlib.Path(data_dir)
  save_csv = pathlib.Path(save_csv)

  print('Load data')
  res = {'pred' : [], 'label' : [], 'pred_wer' : [], 'path' : []}
  dataset = dataset = load_dataset("audiofolder", data_dir=str(data_dir))
  print("set format dataset")
  dataset = dataset['train']
  dataset = dataset.with_format('numpy')

  avoid_OOM()
  dataset = dataset.map(prepare_map_for_eval_transcipt, remove_columns=dataset.column_names, num_proc=3)

  print(dataset)
  index = 0

  for script in pipe(KeyDataset(dataset, "array"), batch_size=batch_size):
    print(script['text'])
    print(dataset[index]['label'])
    print('------------------------')
    res['path'].append(dataset[index]['path'])
    res['pred'].append(script['text'].lower())
    res['label'].append(dataset[index]['label'])
    res['pred_wer'].append(100 * wer.compute(predictions=[script['text'].lower()], references=[dataset[index]['label']]))
    index += 1


  pd.DataFrame.from_dict(res).sort_values(by='pred_wer', ascending=False).to_csv(save_csv, index=False, encoding='utf8')



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model_checkpoint', type=str)
  parser.add_argument('input_path', type=str)
  parser.add_argument('output_path', type=str)
  parser.add_argument('model_version', type=str)
  parser.add_argument('batch_size', type=int)
  args = parser.parse_args()

  check_point = args.model_checkpoint
  input_path = pathlib.Path(args.input_path)
  output_path = pathlib.Path(args.output_path)
  ver = args.model_version
  batch_size = args.batch_size

  # model = WhisperForConditionalGeneration.from_pretrained(r"/mnt/mmlab2024/datasets/final_checkpoint/medium/checkpoint-30001", use_cache=False)
  model = WhisperForConditionalGeneration.from_pretrained(check_point, use_cache=False)


  feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-" + ver)
  tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-" + ver, language="vi", task="transcribe")

  print("Prepare pipe")
  pipe = pipeline(
    "automatic-speech-recognition",
    model = model,
    feature_extractor = feature_extractor,
    tokenizer = tokenizer,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=batch_size,
    return_timestamps=False,
    device='cuda'
  ) 
  print('Start')

  for i in input_path.iterdir():
    print(f"Eval {i} and save at {output_path / (i.name + '.csv')}")
    eval_save_csv(pipe, i, output_path / (i.name + '.csv'), batch_size)