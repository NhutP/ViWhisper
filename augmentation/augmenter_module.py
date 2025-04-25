import librosa
import audiomentations as aumen
from audiomentations import AddBackgroundNoise, PolarityInversion, AddGaussianNoise, AddGaussianSNR, AddShortNoises, BandPassFilter, ClippingDistortion, TanhDistortion, PitchShift, Gain, Compose, RoomSimulator
import soundfile as sf
import pathlib


class augmenter_core:
  def read_mp3(input_path, sampling_rate=16000):
    return librosa.load(input_path, sr=sampling_rate)



class augmenter_module(augmenter_core):
  def __init__(self, background_strogage):
    super().__init__()
    
    self.background_strogage = pathlib.Path(background_strogage)
    self.current_index = 0
    
    self.background_sounds = list(self.background_strogage.rglob('*.wav'))
    self.num_of_background = len(self.background_sounds)
    
    self.Gauss_SNR_high_transformer = AddGaussianSNR(min_snr_db=1.0, max_snr_db=20.0, p=1.0)
    self.Gauss_SNR_average_transformer = AddGaussianSNR(min_snr_db=1.0, max_snr_db=25.0, p=1.0)
    self.Gauss_SNR_low_transformer = AddGaussianSNR(min_snr_db=2.0, max_snr_db=25.0, p=1.0)

    self.BandPass_transformer = BandPassFilter(min_center_freq=100.0, max_center_freq=6000, p=1.0)

    self.ClippingDistortion_transformer = ClippingDistortion(min_percentile_threshold=30, max_percentile_threshold=70)

    self.TanhDistortion_transformer = TanhDistortion(min_distortion=0.2, max_distortion=0.7, p=1.0)

    self.PitchShift_transformer = PitchShift(min_semitones=-2.0, max_semitones=2.0, p=1.0)

    # # just increase volume, should accompanied with another effect
    # self.Gain_up_transformer = Gain(min_gain_db=30, max_gain_db=50, p=1.0)
    # self.Gain_down_transformer = Gain(min_gain_db=-0.5, max_gain_db=0.5, p=1.0)

    self.backgrounds_high_transformers = [AddBackgroundNoise(sounds_path=str(path),\
                                                min_snr_in_db=2.0,\
                                                  max_snr_in_db=10.0,\
                                                      noise_transform=PolarityInversion(),\
                                                        p=1.0) \
                                                          for path in self.background_sounds]  
    self.backgrounds_average_transformers = [AddBackgroundNoise(sounds_path=str(path),\
                                              min_snr_in_db=2.0,\
                                                max_snr_in_db=19.0,\
                                                    noise_transform=PolarityInversion(),\
                                                      p=1.0) \
                                                        for path in self.background_sounds]  
    self.backgrounds_low_transformers = [AddBackgroundNoise(sounds_path=str(path),\
                                          min_snr_in_db=2.0,\
                                            max_snr_in_db=25.0,\
                                                noise_transform=PolarityInversion(),\
                                                  p=1.0) \
                                                    for path in self.background_sounds]  
    
    self.backgrounds_min_transformers = [AddBackgroundNoise(sounds_path=str(path),\
                                          min_snr_in_db=5.0,\
                                            max_snr_in_db=25.0,\
                                                noise_transform=PolarityInversion(),\
                                                  p=1.0) \
                                                    for path in self.background_sounds]  

    # self.room_simulation = [RoomSimulator(min_size_x=, max_size_x= , min_size_y=, max_size_y= ,min_size_z= , max_size_z=,min_absorption_value=, max_absorption_value=, min_target_rt60=)]
    self.room_simulation = [RoomSimulator()]

    # # generate compose of effect
    self.background_Gauss_compose = [Compose([back_low_transformer, self.Gauss_SNR_low_transformer]) for back_low_transformer in self.backgrounds_low_transformers]

    self.background_BandPass_compose = [Compose([back_low_transformer, self.BandPass_transformer]) for back_low_transformer in self.backgrounds_low_transformers]

    self.background_tanhdistortion_compose = [Compose([back_min_transformer, self.TanhDistortion_transformer]) for back_min_transformer in self.backgrounds_min_transformers]

    self.background_PitchShift_compose = [Compose([back_min_transformer, self.PitchShift_transformer]) for back_min_transformer in self.backgrounds_min_transformers]