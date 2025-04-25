import sys
sys.path.insert(0, r'..')

import evaluate
from datasets import load_from_disk, concatenate_datasets
from transformers import WhisperProcessor
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperForConditionalGeneration

from load_data import DataCollatorSpeechSeq2SeqWithPadding

import argparse
import pathlib
import json

from utils.prepare_data import format_string

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
normalizer = BasicTextNormalizer()

global model
global processor


def compute_metrics(pred):
  pred_ids = pred.predictions
  label_ids = pred.label_ids

  # replace -100 with the pad_token_id
  label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

  # we do not want to group tokens when computing the metrics
  pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
  label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

  pred_str = [normalizer(format_string(i)) for i in pred_str]
  label_str = [normalizer(format_string(i)) for i in label_str]

  # pred_str = [normalizer(format_multilingual(i)) for i in pred_str]
  # label_str = [normalizer(format_multilingual(i)) for i in label_str]

  wer = 100 * metric.compute(predictions=pred_str, references=label_str)

  return {"wer": wer}



class trainer_evaluater:
  def __init__(self, processor, model, data_collator, training_args, eval_dataset_folder, dataset_id = ['cmv', 'vivos'], local_data_link=None):
    self.processor = processor
    self.model = model
    self.eval_dataset_folder = pathlib.Path(eval_dataset_folder)
    self.dataset_id = dataset_id
    self.local_data_link = pathlib.Path(local_data_link) if local_data_link else None
    self.data_collator = data_collator
    self.compute_metrics = compute_metrics
    self.training_args = training_args


  def load_from_disk(self):
    data_dict = {}

    print("Load remote datasets")
    for i in self.dataset_id:
      data_dict[i] = load_from_disk(str(self.eval_dataset_folder / i / 'test'))

    # if self.local_data_link:
    #   print("Load local dataset test")
    #   data_dict['test'] = load_from_disk(str(self.local_data_link)).train_test_split(test_size=0.01)['test']

    return data_dict


  def eval(self, output_dir):
    print(f"Start eval on datasets")
    # load dataset and define folder to store checkpoints
    eval_data_dict = self.load_from_disk()

    print("Data info:")
    print(eval_data_dict)

    output_dir = str(output_dir)
    self.training_args.output_dir = output_dir

    # define trainer
    trainer = Seq2SeqTrainer(
      args = self.training_args,
      model = self.model,
      eval_dataset = eval_data_dict,
      data_collator = self.data_collator,
      compute_metrics = self.compute_metrics,
      tokenizer = self.processor.feature_extractor,
    )

    return trainer.evaluate(eval_data_dict)
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('pv', help = 'processor version (tiny, base, small, medium, large)', choices=['tiny', 'base', 'small', 'medium', 'large'], type=str)
  parser.add_argument('preli', help = 'pretrained link, can be local or "openai/whisper-base"', type=str)
  parser.add_argument('out', help='path to root output_dir', type=str)
  
  parser.add_argument('--evdts', help='list of eval dataset (ex: "cmv+vivos")', type=str, default='cmv+vivos')
  parser.add_argument('--evdaf', help='eval dataset folder (which contain sub folders)')
  parser.add_argument('--evibs', help='eval batch size', default=64, type=int)
  
  parser.add_argument('--beam', help='beam size', type=int, default=4)

 
  
  args = parser.parse_args()

  version = args.pv
  pretrain_link = args.preli
  eval_dataset_list = str(args.evdts).split('+')
  remote_eval_data_link = str(args.evdaf)
  local_data_folder = None
  output_dir = str(args.out)
  eval_batch_size = int(args.evibs)
  beam_size = int(args.beam)

  processor = WhisperProcessor.from_pretrained("openai/whisper-" + version, language="vi", task="transcribe")

  model = WhisperForConditionalGeneration.from_pretrained(pretrain_link)

  model.generation_config.language = "vi"
  model.generation_config.num_beams = beam_size

  data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
  metric = evaluate.load("wer")

  training_args = Seq2SeqTrainingArguments(
    output_dir= "",
    per_device_eval_batch_size=eval_batch_size,
    fp16=True,
    evaluation_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_cmv_wer",
    greater_is_better=False,
    push_to_hub=False,

    # jit_mode_eval=use_jit,
    # torch_compile=use_jit,
    # torch_compile_backend=compile_back_end,
    auto_find_batch_size=True
  )

  evaluater = trainer_evaluater(processor, model, data_collator, training_args, remote_eval_data_link, eval_dataset_list, local_data_folder)

  result_link = str(pathlib.Path(output_dir) / (str(pathlib.Path(pretrain_link).name)))

  wer_result = evaluater.eval(result_link)

  with open(result_link + '.json', 'w') as f:
    print(wer_result)
    json.dump(wer_result, f)
    print(f"Result at {result_link + '.json'}")