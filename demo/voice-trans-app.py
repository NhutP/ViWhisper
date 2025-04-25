import streamlit as st
import sounddevice as sd
import soundfile as sf
import librosa
import pathlib
from datetime import datetime
from audio_recorder_streamlit import audio_recorder
#from streamlit_webrtc import webrtc_streamer, WebRtcMode
# import pydub
# import wave
#import time

# from transformers import WhisperProcessor
# from transformers import WhisperForConditionalGeneration
from st_audiorec import st_audiorec
try:
    from transformers import WhisperProcessor
    from transformers import WhisperForConditionalGeneration
    model_loaded = True
except ImportError:
    model_loaded = False

# print(1)

@st.cache_resource()
def load_model(version):
    processor = WhisperProcessor.from_pretrained("openai/whisper-" + version, language="vi", task="transcribe")

    if version == 'tiny':
        model = WhisperForConditionalGeneration.from_pretrained(r"/mnt/mmlab2024/datasets/deploy_checkpoint/tiny_30797")
    if version == 'base':
        model = WhisperForConditionalGeneration.from_pretrained(r"/mnt/mmlab2024/datasets/deploy_checkpoint/base_58976")
    if version == 'small':
        model = WhisperForConditionalGeneration.from_pretrained(r"/mnt/mmlab2024/datasets/deploy_checkpoint/small_42642")
    if version == 'medium':
        model = WhisperForConditionalGeneration.from_pretrained(r"/mnt/mmlab2024/datasets/deploy_checkpoint/medium_30001")

    # if version == 'tiny':
    #     model = WhisperForConditionalGeneration.from_pretrained(r"C:\Users\quang\Desktop\deply_checkpoint\tiny")
    # if version == 'base':
    #     model = WhisperForConditionalGeneration.from_pretrained(r"C:\Users\quang\Desktop\deply_checkpoint\base")
    # if version == 'small':
    #     model = WhisperForConditionalGeneration.from_pretrained(r"C:\Users\quang\Desktop\deply_checkpoint\small")
    # if version == 'medium':
    #     model = WhisperForConditionalGeneration.from_pretrained(r"C:\Users\quang\Desktop\deply_checkpoint\medium")

    return processor, model


def record_audio(filename, duration=5):
    fs = 16000  # Sample rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    sf.write(filename, recording, fs)


def transcribe_audio(filename, model_version):
    audio_array, sr = librosa.load(filename, sr=16000)
    processor, model = load_model(model_version)

    input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features

    model.to('cpu')
    predicted_ids = model.generate(input_features)
    model.to('cpu')

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0].lower()



def main():
    st.title("Vietnamese Speech to Text")

    # st.sidebar.header("Recording Controls")
    # duration = st.sidebar.slider("Recording Duration (seconds)", min_value=1, max_value=30, value=5)
    # record_button = st.sidebar.button("Record")

    model_version = st.sidebar.radio("Choose Model Version", ["tiny", "base", "small", "medium"])
    st.warning("This demo only uses cpu, so the the inference time of large models can take up to minutes")
    
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.success("Recording complete!")

    transcript_button = st.button('Transcribe')
    
    if transcript_button:
        now = datetime.now()
        filename = pathlib.Path(str(r"/home/mmlab/s2t_history/speech/"  +  str(now.strftime("%Y%m%d%H%M%S") + '_' + model_version) + '.wav'))
        with open(filename, 'wb') as w:
            w.write(wav_audio_data)

        st.info("Transcribing with CPU, wait for a while...")
        transcript = transcribe_audio(filename, model_version)

        with open('/home/mmlab/s2t_history/text/' + filename.stem + '.txt', 'w', encoding='utf8') as w:
            w.write(transcript)
        st.write("Transcription:")
        st.write(transcript)

    # if record_button:
    #     st.info(f"Recording {duration} seconds of audio...")
    #     now = datetime.now()
    #     filename = pathlib.Path(str('/home/mmlab/s2t_history/speech/'  +  str(now.strftime("%Y%m%d%H%M%S") + '_' + model_version) + '.wav'))
    #     # record_audio(filename, duration)
    #     # st.success("Recording complete!")
    #     # st.audio(str(filename), format='audio/wav')

    #     st.info("Transcribing...")
    #     transcript = transcribe_audio(filename, model_version)

    #     with open('/home/mmlab/s2t_history/text/' + filename.stem + '.txt', 'w', encoding='utf8') as w:
    #         w.write(transcript)
    #     st.write("Transcription:")
    #     st.write(transcript)

    # if record_button:
    #     st.info(f"Recording {duration} seconds of audio...")
    #     now = datetime.now()
    #     filename = pathlib.Path(str(r'C:\Users\quang\Desktop\ssss\speech\\'  +  str(now.strftime("%Y%m%d%H%M%S") + '_' + model_version) + '.wav'))
        
    #     record_audio(str(filename), duration)
    #     st.success("Recording complete!")

    #     st.audio(str(filename), format='audio/wav')

    #     st.info("Transcribing...")
    #     transcript = transcribe_audio(filename, model_version)

    #     with open(r'C:\Users\quang\Desktop\ssss\text\\' + filename.stem + '.txt', 'w', encoding='utf8') as w:
    #         w.write(transcript)
    #     st.write("Transcription:")
    #     st.write(transcript)

if __name__ == "__main__":
    main()






# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
# from io import BytesIO
# import librosa
# import numpy as np
# import soundfile as sf
# import time

# try:
#     from transformers import WhisperProcessor
#     from transformers import WhisperForConditionalGeneration
#     model_loaded = True
# except ImportError:
#     model_loaded = False


# @st.cache_resource()
# def load_model(version):
#     processor = WhisperProcessor.from_pretrained("openai/whisper-" + version, language="vi", task="transcribe")
#     model_path = {
#         'tiny': r"C:\Users\quang\Desktop\deply_checkpoint\tiny",
#         'base': r"C:\Users\quang\Desktop\deply_checkpoint\base",
#         'small': r"C:\Users\quang\Desktop\deply_checkpoint\small",
#         'medium': r"/mnt/mmlab2024/datasets/deploy_checkpoint/medium_30001"
#     }
#     model = WhisperForConditionalGeneration.from_pretrained(model_path[version])
#     return processor, model

# class AudioProcessor(AudioProcessorBase):
#     def __init__(self):
#         self.audio_data = []
#         self.start_time = None

#     def recv(self, frame):
#         if self.start_time is None:
#             self.start_time = time.time()
#         elapsed_time = time.time() - self.start_time
#         if elapsed_time > max_recording_duration:
#             webrtc_ctx.stop()
#             return None
#         audio = frame.to_ndarray()
#         self.audio_data.append(audio.tobytes())
#         return frame

# def transcribe_audio(audio_data, model_version):
#     audio_array = np.frombuffer(audio_data, dtype=np.float32)
#     processor, model = load_model(model_version)
#     input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
#     predicted_ids = model.generate(input_features)
#     transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
#     return transcription[0].lower()

# def save_and_resample_audio(audio_data, filename, target_sr=16000):
#     # Convert byte data to float32 numpy array
#     audio_array = np.frombuffer(audio_data, dtype=np.float32)
#     # Resample audio to the target sample rate
#     resampled_audio = librosa.resample(audio_array, orig_sr=48000, target_sr=target_sr)
#     # Save the resampled audio to a file
#     sf.write(filename, resampled_audio, target_sr)

# def main():
#     global max_recording_duration, webrtc_ctx

#     st.title("Voice Recorder & Transcriber")

#     st.sidebar.header("Recording Controls")
#     max_recording_duration = st.sidebar.slider("Recording Duration (seconds)", min_value=1, max_value=30, value=5)
#     model_version = st.sidebar.radio("Choose Model Version", ["tiny", "base", "small", "medium"])

#     webrtc_ctx = webrtc_streamer(
#         key="audio-recorder",
#         mode=WebRtcMode.SENDONLY,
#         media_stream_constraints={"audio": True, "video": False},
#         audio_processor_factory=AudioProcessor,
#     )

#     if st.button("Start Recording"):
#         if webrtc_ctx.state.playing:
#             st.warning("Recording is already in progress.")
#         else:
#             st.info("Click the 'Stop Recording' button to save and transcribe the audio.")

#     if st.button("Stop Recording"):
#         if webrtc_ctx.state.playing:
#             webrtc_ctx.stop()
#             st.success("Recording stopped.")

#             if webrtc_ctx.audio_processor:
#                 audio_processor = webrtc_ctx.audio_processor
#                 audio_data = b"".join(audio_processor.audio_data)
#                 temp_filename = "temp.wav"
#                 save_and_resample_audio(audio_data, temp_filename, target_sr=16000)
#                 st.audio(temp_filename, format="audio/wav")

#                 st.info("Transcribing...")
#                 transcript = transcribe_audio(audio_data, model_version)
#                 st.write("Transcription:")
#                 st.write(transcript)
#             else:
#                 st.warning("Please start the recording first.")
#         else:
#             st.warning("Recording is not in progress.")

# if __name__ == "__main__":
#     main()