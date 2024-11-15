# ViWhisper
## Introduction
- We release a new model for Vietnamese speech regconition task.
- We fine-tuned openai/whisper on our new dataset [VSV-1100](https://huggingface.co/datasets/NhutP/VSV-1100).

## Training data

| [VSV-1100](https://huggingface.co/datasets/NhutP/VSV-1100) | T2S* | [CMV14-vi](https://huggingface.co/datasets/mozilla-foundation/common_voice_14_0) |[VIVOS](https://huggingface.co/datasets/AILAB-VNUHCM/vivos)| [VLSP2021](https://vlsp.org.vn/index.php/resources) | Total|
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|   1100 hours  |   11  hours |   3.04 hours  |    13.94  hours| 180 hours | 1308 hours |

\* We use a text-to-speech model to generate sentences containing words that do not appear in our dataset.

## WER result
|Version| [CMV14-vi](https://huggingface.co/datasets/mozilla-foundation/common_voice_14_0) | [VIVOS](https://huggingface.co/datasets/AILAB-VNUHCM/vivos) | [VLSP2020-T1](https://vlsp.org.vn/index.php/resources) | [VLSP2020-T2](https://vlsp.org.vn/index.php/resources) | [VLSP2021-T1](https://vlsp.org.vn/index.php/resources) | [VLSP2021-T2](https://vlsp.org.vn/index.php/resources) |[Bud500](https://huggingface.co/datasets/linhtran92/viet_bud500) |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|Tiny|Updating|Updating|Updating|Updating|Updating|Updating|Updating|
|Base|Updating|Updating|Updating|Updating|Updating|Updating|Updating|
|[Small](https://huggingface.co/NhutP/ViWhisper-small)|9.79|5.74|14.15|39.25| 14 | 10.06 | 5.97 |
|Medium|Updating|Updating|Updating|Updating|Updating|Updating|Updating|


## Usage
### Inference
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
model_id = .. # "NhutP/ViWhisper-small" or "NhutP/ViWhisper-tiny", ...
# load model and processor
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.config.forced_decoder_ids = None
# load a sample
array, sampling_rate = librosa.load(...) # Load some 
input_features = processor(array, sampling_rate=sampling_rate, return_tensors="pt").input_features 
# generate token ids
predicted_ids = model.generate(input_features)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
```
### Use with pipeline
```python
from transformers import pipeline
model_id = .. # "NhutP/ViWhisper-small" or "NhutP/ViWhisper-tiny", ...
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    max_new_tokens=128,
    chunk_length_s=30,
    return_timestamps=False,
    device= '...' # 'cpu' or 'cuda'
) 
output = pipe(path_to_audio_samplingrate_16000)['text']
```

## Citation

```
@misc{VSV-1100,
    author = {Pham Quang Nhut and Duong Pham Hoang Anh and Nguyen Vinh Tiep},
    title = {VSV-1100: Vietnamese social voice dataset},
    url = {https://github.com/NhutP/VSV-1100},
    year = {2024}
}
```

Contact me at: 22521061@gm.uit.edu.vn (Pham Quang Nhut)
