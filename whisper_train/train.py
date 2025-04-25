import sys
sys.path.insert(0, r'..')

from transformers import WhisperProcessor
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperForConditionalGeneration
from transformers import AdamW

from safetensors.torch import load_model, save_model, load_file

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR

from datasets import load_dataset, load_from_disk, concatenate_datasets
import evaluate

from load_data import whisper_data_loader

import pathlib
import time
from datetime import datetime
import psutil
import json
import argparse
import sys
from typing import Any, Dict, List, Union
import random
from load_data import DataCollatorSpeechSeq2SeqWithPadding
import transformers

from tqdm import tqdm
from utils.prepare_data import format_string

assert torch.cuda.is_available()

global processor
global model
global metric

class whisper_trainer(whisper_data_loader):
  def __init__(self, data_folder, seq2seq_arguments, data_collator, eval_dataset_list, compute_metrics, processor, model, previous_checkpoint):
    super().__init__(data_folder)
    self.training_args = seq2seq_arguments
    self.eval_dataset_list = eval_dataset_list
    self.data_collator=data_collator
    self.compute_metrics=compute_metrics
    self.model = model
    self.processor = processor
    self.previous_checkpoint = previous_checkpoint

  def train(self, root_output_dir, optimizer_and_scheduler, use_local_train=True, use_remote_train = True, use_remote_validate = True, remote_data_strogage = None, split_local_data=True, keep_dataset_in_memory = False, flatten_indices=False):
    print(f"Start training on dataset")
    # load dataset and define folder to store checkpoints
    # now = datetime.now()
    # output_dir = str(root_output_dir  /  now.strftime("%Y%m%d%H%M%S"))

    self.training_args.output_dir = root_output_dir
    ###
    # load the train dataset
    if use_local_train:
      train_dataset = self.load_all()

    if use_remote_train:
      remote_train_data = self.load_remote_train(remote_data_strogage, data_id=self.eval_dataset_list, keep_dataset_in_memory=keep_dataset_in_memory)
      if use_local_train:
        train_dataset = concatenate_datasets([train_dataset, remote_train_data])
      else:
        train_dataset = remote_train_data

    # load the eval dataset
    # if use remote dataset, load them, if not, define an empty dict
    if use_remote_validate:
      print("Load remote eval dataset")
      eval_dataset = self.load_remote_eval(remote_data_strogage, data_id=self.eval_dataset_list, keep_dataset_in_memory=keep_dataset_in_memory)
    else:
      eval_dataset = {}

    # if split local data, then merge with the remote validation data
    if split_local_data:
      print("Split local data")
      split_dataset = train_dataset.train_test_split(test_size=0.01)
      eval_dataset['test'] = split_dataset['test']
      train_dataset = split_dataset['train']
      
    if flatten_indices:
      train_dataset = train_dataset.flatten_indices()

    print("Train dataset info:")
    print(train_dataset)
    
    print("Eval dataset info:")
    print(eval_dataset)

    if  optimizer_and_scheduler[0] is None or optimizer_and_scheduler[1] is None:
      trainer = Seq2SeqTrainer(
      args = self.training_args,
      model = self.model,
      train_dataset = train_dataset,
      eval_dataset = eval_dataset,
      data_collator = self.data_collator,
      compute_metrics = self.compute_metrics,
      tokenizer = self.processor.feature_extractor
      )
    else:
    # define trainer
      trainer = Seq2SeqTrainer(
        args = self.training_args,
        model = self.model,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = self.data_collator,
        compute_metrics = self.compute_metrics,
        tokenizer = self.processor.feature_extractor,
        optimizers =  optimizer_and_scheduler
      )
    if self.previous_checkpoint is not None:
      print("Continue from checkpoint")
    trainer.train(resume_from_checkpoint = self.previous_checkpoint)



if __name__ == '__main__':
  default_lr = {'tiny' : 3.75e-5, 'base': 2.5e-5, 'small' : 1.25e-5, 'medium' : 6.25e-6, 'large' : 5e-6}
  default_min_lr = {'tiny' : 3.75e-7, 'base': 2.5e-7, 'small' : 1.25e-7, 'medium' : 6.25e-8, 'large' : 5e-8}

  # default_lr = {'tiny' : 1.875e-05, 'base': 2.5e-5, 'small' : 6.25e-06, 'medium' :  3.125e-06, 'large' : 5e-6}

  parser = argparse.ArgumentParser()

  parser.add_argument('pv', help = 'processor version (tiny, base, small, medium, large)', choices=['tiny', 'base', 'small', 'medium', 'large'], type=str)
  parser.add_argument('preli', help = 'pretrained link, can be local or "openai/whisper-base"', type=str)
  parser.add_argument('da', help='path to processed processed data folder, if multiple, split by the "+"', type=str)
  parser.add_argument('out', help='path to root output_dir', type=str)

  parser.add_argument('--rmstrda', help='path to remote data strogage', type=str)
  parser.add_argument('--loda', help='use local train data', default='y', choices=['y', 'n'], type=str)
  parser.add_argument('--slda', choices=['y', 'n'], help='whether to split local data to validation', default='n',type=str)
  parser.add_argument('--revali', help='use remote validation data', default='y', choices=['y', 'n'], type=str)
  parser.add_argument('--remtra', help='use remote train data', default='y', choices=['y', 'n'], type=str)

  parser.add_argument('--nte', default = 5, help='num train epoch', type=int)
  parser.add_argument('--evt', default = 0.025, help='eval steps', type=float)
  parser.add_argument('--sat', default = 0.025, help='save steps', type=float)
  parser.add_argument('--lr', help='learning rate, leave blank to use default', type=str)

  parser.add_argument('--ibs', default = 64, help='intitial batch size', type=int)
  parser.add_argument('--jit', choices=['y', 'n'], help='use JIT compile or not', default='n', type=str)

  parser.add_argument('--evdts', help='list of names of remote dataset (ex: "cmv+vivos+bud500")', type=str, default='cmv+vivos+bud500')
  parser.add_argument('--dtsbm', help='dataset for best model', default='cmv14vivos', type=str)

  parser.add_argument('--jitbe', help='compile backend', type=str, default='eager')
  parser.add_argument('--wu', help='warm up ratio', type=float, default=0.05)
  parser.add_argument('--kime', choices=['y', 'n'], help='whether to keep dataset in memory, only use when have large memory', default='n', type=str)
  parser.add_argument('--ftid', choices=['y', 'n'], help='whether to flatten indices, only use when have large memory', default='n', type=str)
  parser.add_argument('--evibs', help='eval batch size', default=64, type=int)
  parser.add_argument('--accgrad', help='gradient accumulation', default=1, type=int)
  parser.add_argument('--device', help='device, default is cuda', default='cuda', type=str)
  parser.add_argument('--ctn', help = 'continue from checkpoint, path to previous checkpoint to start from', type=str)
  args = parser.parse_args()

  version = args.pv
  model_id = args.preli
  data_folders = pathlib.Path(args.da)
  out_dir = pathlib.Path(args.out)
  learning_rate = float(args.lr) if args.lr is not None else default_lr[version]

  remote_data_strogage = pathlib.Path(args.rmstrda) if args.rmstrda is not None else None
  split_local_data = True if str(args.slda).lower().strip() == 'y' else False
  use_local_train = True if str(args.loda).lower().strip() == 'y' else False
  remote_data_validate = True if str(args.revali).lower().strip() == 'y' else False
  use_remote_train = True if str(args.remtra).lower().strip() == 'y' else False
  eval_dataset_list = str(args.evdts).split('+')
  dataset_bestmodel = str(args.dtsbm)

  flatten_indices = True if str(args.ftid).lower().strip() == 'y' else False
  keep_dataset_in_memory = True if str(args.kime).lower().strip() == 'y' else False

  num_epoch = int(args.nte)
  eval_step = float(args.evt)
  save_step = float(args.sat)

  initial_batch_size = int(args.ibs)
  use_jit = True if str(args.jit).strip().lower() == 'y' else False
  compile_back_end = str(args.jitbe)
  warm_up = float(args.wu)

  device = str(args.device)

  eval_batch_size = int(args.evibs)
  acc_grad = int(args.accgrad)
  previous_checkpoint = args.ctn

  # load pretrained model and processor
  processor = WhisperProcessor.from_pretrained("openai/whisper-" + version, language="vi", task="transcribe")

  model = WhisperForConditionalGeneration.from_pretrained(model_id, use_cache=False)
  model.gradient_checkpointing_enable()
  # model = model.to(device)
  model.config.dropout = 0.1
  model.generation_config.language = "vi"
  

  # # create optimizer
  optimizer = AdamW(model.parameters(), lr = learning_rate)
  scheduler = LinearLR(optimizer, )
  # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True, min_lr=learning_rate / 100, eps=1e-10)

  # create optimizer
  optimizer = None
  scheduler = None

  data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
  metric = evaluate.load("wer")


  # WER compute
  def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [format_string(i) for i in pred_str]
    label_str = [format_string(i) for i in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)


    return {"wer": wer}
  

  # define training argument
  training_args = Seq2SeqTrainingArguments(
    output_dir= "",
    per_device_train_batch_size=initial_batch_size,
    gradient_accumulation_steps=acc_grad,  # increase by 2x for every 2x decrease in batch size
    learning_rate=learning_rate,
    warmup_ratio=warm_up,
    num_train_epochs=num_epoch,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=eval_batch_size,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=save_step,
    eval_steps=eval_step,
    logging_steps=100,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model= 'eval_' + dataset_bestmodel + "_wer",
    greater_is_better=False,
    push_to_hub=False,
    ddp_backend='nccl',
    save_safetensors=False
    # optim="adamw_bnb_8bit"
    
    # jit_mode_eval=use_jit,
    # torch_compile=use_jit,
    # torch_compile_backend=compile_back_end,
    # auto_find_batch_size=True
  )

  # if use JIT compile
  if use_jit:
    print('Use JIT')
    torch._dynamo.config.suppress_errors = True
    TORCH_LOGS="+dynamo"
    TORCHDYNAMO_VERBOSE=1
    training_args.jit_mode_eval = True
    training_args.torch_compile=use_jit
    training_args.torch_compile_backend=compile_back_end
  
  
  whisper_train = whisper_trainer(data_folders, training_args, data_collator, eval_dataset_list, compute_metrics, processor, model, previous_checkpoint)


  whisper_train.train(out_dir, (optimizer, scheduler), use_local_train=use_local_train, split_local_data=split_local_data, use_remote_validate= remote_data_validate, use_remote_train=use_remote_train,remote_data_strogage=remote_data_strogage, keep_dataset_in_memory=keep_dataset_in_memory, flatten_indices=flatten_indices)