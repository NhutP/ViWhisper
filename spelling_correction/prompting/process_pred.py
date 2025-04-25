import sys
sys.path.insert(0, r'..')

from tqdm import tqdm
import numpy as np
import pandas as pd
import google.generativeai as genai

import argparse
import time
import pathlib
import pandas as pd


import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

from utils.prepare_data import format_string
import evaluate
from datetime import datetime

wer = evaluate.load("wer")

class gemini_pred_processor:
  def __init__(self,gemini_api_key=None):
    self.gemini_api_key = gemini_api_key

  def process_single_sentence(self, sentence):
    model = genai.GenerativeModel('gemini-pro')

    # prompt = f'sửa lỗi chính tả cho câu "{sentence}", chỉ ghi ra kết quả không giải thích gì thêm, chỉ sửa lỗi ở những từ có khả năng cao bị nhầm lẫn do phát âm giống nhau (ví dụ: (về, dề), (tín ,tính), ....) hoặc sai dấu câu ((ngả, ngã), ...)), không sửa bằng cách thay từ đồng nghĩa không có phát âm giống nhau (ví dụ: không thay "vứt" thành "bỏ", "cốc" thành "ly" vì các cặp từ này chỉ có cùng nghĩa mà không có phát âm tương đồng), nếu không có lỗi hoặc không có từ có phát âm giống nhau để sửa thì giữ nguyên, chỉ sửa lỗi chính tả, không chèn, xóa từ. Ví dụ: câu "đèn bị hư gò ba" sửa thành "đèn bị hư rồi ba" vì các từ "gò" và "rồi" đôi khi có phát âm giống nhau, câu "nhưng tình yêu vẫn nồng nàn và say đấm như buổi ban đầu" sửa thành "nhưng tình yêu vẫn nồng nàn và say đắm như buổi ban đầu" vì 2 từ đấm và đắm có phát âm giống nhau, câu "chị ngả em nâng" sửa thành "chị ngã em nâng" vì từ "ngả" bị sai dấu câu, còn câu "xuân này em không về" thì giữ nguyên vì không có lỗi chính tả.'

    # prompt = f'câu này viết sai chính tả, hãy sửa lại: "{sentence}"'

    prompt = f"""
    Sửa lỗi chính tả do phiên âm tiếng Việt. Hạn chế sửa từ đồng nghĩa và thay đổi ít ký tự nhất có thể. Dưới đây là một số ví dụ.

    thu chạm ngỏ rất hiền: thu chạm ngõ rất hiền
    rồi mặc dù biết là ngớ ngận tôi cứ nói ra: rồi mặc dù biết là ngớ ngẩn tôi cứ nói ra
    các chương trình ca nhạc phần lớn đều không đột phé: các chương trình ca nhạc phần lớn đều không đột phá
    {sentence}: """

    response = model.generate_content(prompt)
    return response.text


  def fix_all(self, csv_file):
    API_index = 0

    data = {}
    df = pd.read_csv(csv_file)

    for col in df.columns:
      data[col] = df[col].tolist()

    data['pred_with_gemini'] = []
    data['is_changed_gemini'] = []
    
    data_len = len(data['pred'])
    failed_key = 0
    for i in range(data_len):
      while True:
        try:
          processed_pred = self.process_single_sentence(data['pred'][i])
          print(f"pred: {data['pred'][i]}")
          print(f"pred with gemini: {format_string(processed_pred.lower())}")
          print(f"label: {data['label'][i].lower()}")

          if len(format_string(processed_pred)) == len(data['pred'][i]) and wer.compute(references=[data['label'][i]], predictions=[format_string(processed_pred)]) < 0.2 :
            data['pred_with_gemini'].append(format_string(processed_pred))
            data['is_changed_gemini'].append(1 if format_string(processed_pred) != data['pred'][i] else 0)
          else:
            data['pred_with_gemini'].append(data['pred'][i])
            data['is_changed_gemini'].append(0)

          API_index = (API_index + 1) % len(self.gemini_api_key)
          failed_key = 0
          print('---------------------------------')
          break
        except Exception as e:
          print(f"Exception: {e}")
          # if "429" in str(e):
          #   API_index = (API_index + 1) % len(self.gemini_api_key)
          #   genai.configure(api_key = self.gemini_api_key[API_index])
          #   print("change API key")
          #   failed_key += 1
          #   time.sleep(10)
          
          # elif "Your default credentials were not found." in str(e):
          API_index = (API_index + 1) % len(self.gemini_api_key)
          genai.configure(api_key = self.gemini_api_key[API_index])
          print("change API key")
          failed_key += 1
          time.sleep(10)
          print('**********************************')

          if failed_key >= len(self.gemini_api_key) + 1:
            print("Enough failed key")
            print(datetime.now().strftime("%H%M%S"))
            time.sleep(3600)
            failed_key = 0
      time.sleep(10)
    data['wer_gemini'] = [100 * wer.compute(references=[data['label'][i]], predictions=[data['pred_with_gemini'][i]]) for i in range(len(data['pred']))]
    pd.DataFrame.from_dict(data).sort_values(by='wer_gemini', ascending=False).to_csv(csv_file, index=False)


class phogpt_pred_processor:
  def __init__(self):
    model_path = "vinai/PhoGPT-4B-Chat" 
    self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  
    self.config.init_device = "cuda"
    # config.attn_config['attn_impl'] = 'flash' # If installed: this will use either Flash Attention V1 or V2 depending on what is installed

    self.model = AutoModelForCausalLM.from_pretrained(model_path, config=self.config, torch_dtype=torch.bfloat16, trust_remote_code=True, cache_dir = '/mnt/mmlab2024/datasets/VNSTT/temp_cache/cache1').to('cuda')
    # If your GPU does not support bfloat16:
    # model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
    self.model.eval()  

    self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir = '/mnt/mmlab2024/datasets/VNSTT/temp_cache/cache1')  


  def process_single_sentence(self, sentence):
    # prompt = f'### Câu hỏi: Sửa lỗi chính tả (chỉ ghi ra kết quả không giải thích gì thêm, chỉ sửa lỗi ở những từ có khả năng cao bị nhầm lẫn do phát âm giống nhau (ví dụ: (về, dề), (tín ,tính), ....) hoặc sai dấu câu ((ngả, ngã), ...)),, nếu không có lỗi hoặc không có từ có phát âm giống nhau để sửa thì giữ nguyên, chỉ sửa lỗi chính tả, không chèn, xóa từ. Ví dụ: câu "đèn bị hư gò ba" sửa thành "đèn bị hư rồi ba" vì các từ "gò" và "rồi" đôi khi có phát âm giống nhau, câu "nhưng tình yêu vẫn nồng nàn và say đấm như buổi ban đầu" sửa thành "nhưng tình yêu vẫn nồng nàn và say đắm như buổi ban đầu" vì 2 từ đấm và đắm có phát âm giống nhau, câu "chị ngả em nâng" sửa thành "chị ngã em nâng" vì từ "ngả" bị sai dấu câu, còn câu "xuân này em không về" thì giữ nguyên vì không có lỗi chính tả:\n "{sentence}"\n### Trả lời:'
    # prompt = f"### Câu hỏi: Sửa lỗi chính tả:\n {sentence}\n### Trả lời:"  
    prompt = f'### Câu hỏi: câu này viết sai chính tả, hãy sửa lại: "{sentence}"\n### Trả lời:'
    # prompt = f'câu này viết sai chính tả, hãy sửa lại: "{sentence}"'
    input_ids = self.tokenizer(prompt, return_tensors="pt")  

    outputs = self.model.generate(  
    inputs=input_ids["input_ids"].to("cuda"),  
    attention_mask=input_ids["attention_mask"].to("cuda"),  
    do_sample=True,  
    temperature=1.0,  
    top_k=50,  
    top_p=0.9,  
    max_new_tokens=1024,  
    eos_token_id=self.tokenizer.eos_token_id,  
    pad_token_id=self.tokenizer.pad_token_id  
    )  

    response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  
    response = response.split("### Trả lời:")[1]
    return response


  def fix_all(self, csv_file):
    data = {}
    df = pd.read_csv(csv_file)
    
    # data['pred'] = [data['pred'][i] for i in range(len(data['pred']))]
    # data['pred_with_phogpt'] = []
    # data['label'] = [data['label'][i] for i in range(len(data['pred']))]
    # data['is_changed'] = []
    # data['path'] = [data['path'][i] for i in range(len(data['pred']))]

    for col in df.columns:
      data[col] = df[col].tolist()
    
    data['pred_with_phogpt'] = []
    data['is_changed_phogpt'] == []

    for i in range(len(data['pred'])):
      processed_pred = self.process_single_sentence(data['pred'][i])
      print(f"pred: {data['pred'][i]}")
      print(f"pred with phogpt: {format_string(processed_pred.lower())}")
      print(f"label: {data['label'][i].lower()}")
      print('--------------------------------------------')
      if len(format_string(processed_pred)) == len(data['pred'][i]) and wer.compute(references=[data['label'][i]], predictions=[format_string(processed_pred)]) < 0.2 :
        data['pred_with_phogpt'].append(format_string(processed_pred))
        data['is_changed_phogpt'].append(1 if format_string(processed_pred) != data['pred'][i] else 0)
      else:
        data['pred_with_phogpt'].append(data['pred'][i])
        data['is_changed_phogpt'].append(0)

    data['wer_phogpt'] = [100 * wer.compute(references=[data['label'][i]], predictions=[data['pred_with_phogpt'][i]]) for i in range(len(data['pred']))]
    pd.DataFrame.from_dict(data).sort_values(by='wer_phogpt', ascending=False).to_csv(csv_file, index=False)



class bartpho_processor:
  def __init__(self):
    self.corrector = pipeline("text2text-generation", model= "bmd1905/vietnamese-correction", device='cuda')  


  def process_single_sentence(self, sentence):
    prediction = self.corrector([sentence], max_length=512)
    return prediction[0]['generated_text']


  def fix_all(self, csv_file):
    data = {}
    df = pd.read_csv(csv_file)

    for col in df.columns:
      data[col] = df[col].tolist()
    
    data['pred_with_spell_correction'] = []
    data['is_changed_spell_correction'] = []
    
    for i in range(len(data['pred'])):
      processed_pred = self.process_single_sentence(data['pred'][i])
      print(f"pred: {data['pred'][i]}")
      print(f"pred with spell_correction: {format_string(processed_pred.lower())}")
      print(f"label: {data['label'][i].lower()}")
      print('--------------------------------------------')
      if len(format_string(processed_pred)) == len(data['pred'][i]) and wer.compute(references=[data['label'][i]], predictions=[format_string(processed_pred)]) < 0.2 :
        data['pred_with_spell_correction'].append(format_string(processed_pred))
        data['is_changed_spell_correction'].append(1 if format_string(processed_pred) != data['pred'][i] else 0)
      else:
        data['pred_with_spell_correction'].append(data['pred'][i])
        data['is_changed_spell_correction'].append(0)

    data['wer_spell_correction'] = [100 * wer.compute(references=[data['label'][i]], predictions=[data['pred_with_spell_correction'][i]]) for i in range(len(data['pred']))]
    pd.DataFrame.from_dict(data).sort_values(by='wer_spell_correction', ascending=False).to_csv(csv_file, index=False)



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('inpa', type=str, help='The path to the input file or directory')
  parser.add_argument('lm', type=str, help='phogpt or gemini or bartpho')
  args = parser.parse_args()

  input_path = pathlib.Path(args.inpa)
  lm = args.lm

  if lm == 'gemini':
    with open("gemini_api_key.txt", 'r') as r:
      gemini_api_key = r.read().split('\n')
    processor = gemini_pred_processor(gemini_api_key)

  elif lm == 'phogpt':
    processor = phogpt_pred_processor()

  elif lm == 'bartpho':
    processor = bartpho_processor()


  for i in input_path.rglob('*.csv'):
    print(f"Post processing at {i}")
    processor.fix_all(i)