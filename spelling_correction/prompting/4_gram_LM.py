import sys
sys.path.insert(0, r'..')



from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


import whisper
from utils.prepare_data import format_string
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
normalizer = BasicTextNormalizer()




processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
# model = WhisperForConditionalGeneration.from_pretrained("/mnt/mmlab2024/datasets/final_checkpoint/tiny_only_da1_single_tts/checkpoint-27816")
model1 = WhisperForConditionalGeneration.from_pretrained("/mnt/mmlab2024/datasets/final_checkpoint/medium/checkpoint-30001").to('cuda')
model2 = whisper.load_model("/mnt/mmlab2024/datasets/final_checkpoint/pt_format/checkpoint-30001.pt")

while True:
  file_path = input("input: ")
  model1.config.forced_decoder_ids = None

  array, _ = librosa.load(file_path, sr=16000)

  input_features = processor(array, sampling_rate=16000, return_tensors="pt").input_features
  predicted_ids = model1.generate(input_features.to('cuda'))
  transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
  print("Prediction no LM")
  print(format_string(transcription[0]))

  ##################

  # model = whisper.load_model("/mnt/mmlab2024/datasets/final_checkpoint/pt_format/checkpoint-27816_da1_tts.pt")
  # model = whisper.load_model("/mnt/mmlab2024/datasets/final_checkpoint/pt_format/checkpoint-34599.pt")



  # load audio and pad/trim it to fit 30 seconds
  audio = whisper.load_audio(file_path)
  audio = whisper.pad_or_trim(audio)

  # make log-Mel spectrogram and move to the same device as the model
  mel = whisper.log_mel_spectrogram(audio).to(model2.device)
  # detect the spoken language
  _, probs = model2.detect_language(mel)
  print(f"Detected language: {max(probs, key=probs.get)}")

  # decode the audio
  options = whisper.DecodingOptions(withlm=True, lm_path=r"/mnt/mmlab2024/datasets/ViWhisper/vi_lm_4grams.bin", beam_size=5, without_timestamps=True, lm_alpha=1, lm_beta=0.5, language='vi')
  # options = whisper.DecodingOptions(without_timestamps=True, language='vi')


  result = whisper.decode(model2, mel, options)
  print("Prediction with LM")
  print(format_string(normalizer(result.text)))