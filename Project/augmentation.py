import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os


# Total 41 augmentations available in audiomentations library
from audiomentations import (AddGaussianNoise, AddBackgroundNoise, AddColorNoise, AddGaussianSNR, AddShortNoises, 
                             AdjustDuration, AirAbsorption, Aliasing, ApplyImpulseResponse, BandPassFilter, 
                             BandStopFilter, BitCrush, Clip, ClippingDistortion, Gain, GainTransition, 
                             HighPassFilter, HighShelfFilter, Lambda, Limiter, LoudnessNormalization, 
                             LowPassFilter, LowShelfFilter, Mp3Compression, Normalize, Padding, PeakingFilter, 
                             PitchShift, PolarityInversion, RepeatPart, Resample, Reverse, RoomSimulator, 
                             SevenBandParametricEQ, Shift, SpecChannelShuffle, SpecFrequencyMask, TanhDistortion, 
                             TimeMask, TimeStretch, Trim)

# Function to load audio
def load_audio(input_file):
    return librosa.load(input_file, sr=None)

# Function to save audio
def save_audio(output_file, y, sr):
    sf.write(output_file, y, sr)

# Function to plot waveform and mel spectrogram
def plot_waveform_and_mel_spectrogram(audio, sr, title=""):
    plt.figure(figsize=(8, 4))
    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title(f"Waveform: {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot mel spectrogram
    plt.subplot(2, 1, 2)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, fmax=8000)
    
    plt.tight_layout()

# * Individual augmentation functions, could use chain transform as well
# * P stands for probability of applying the augmentation, 1 means always apply

# 1. Add Gaussian Noise
def add_gaussian_noise(input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    augment = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="Add Gaussian Noise")

    plt.show()

# 2. Time Stretch
def time_stretch(input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    augment = TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="Time Stretch")

    plt.show()

# 3. Pitch Shift
def pitch_shift(input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    augment = PitchShift(min_semitones=-2, max_semitones=2, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="Pitch Shift")

    plt.show()

# 4. Gain
def gain(input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    augment = Gain(min_gain_in_db=-12, max_gain_in_db=12, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="Gain")

    plt.show()

# 5. Loudness Normalization
def loudness_normalization(input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    augment = LoudnessNormalization(target_loudness_in_db=-20, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="Loudness Normalization")

    plt.show()

# 6. High Pass Filter
def high_pass_filter(input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    augment = HighPassFilter(min_cutoff_frequency=500, max_cutoff_frequency=2000, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="High Pass Filter")

    plt.show()

# 7. Low Pass Filter
def low_pass_filter(input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    augment = LowPassFilter(min_cutoff_frequency=500, max_cutoff_frequency=2000, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="Low Pass Filter")

    plt.show()

# 8. Trim
def trim(input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    augment = Trim(top_db=20, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="Trim")

    plt.show()

# 9. Resample
def resample(input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    augment = Resample(min_sample_rate=8000, max_sample_rate=44100, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="Resample")

    plt.show()

# 10. Room Simulator
def room_simulator(input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    augment = RoomSimulator(room_preset='livingroom', p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="Room Simulator")

    plt.show()

# 11. Apply Impulse Response
def apply_impulse_response(input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    augment = ApplyImpulseResponse(impulse_response_path='impulse_responses', p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="Apply Impulse Response")

    plt.show()

# cari paper biasa orang time shift berapa, di git atau ga di medium.
def shift (input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    augment = Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="Shift")

    plt.show()

# Mapping method names to function calls
augmentations = {
    "add_gaussian_noise": add_gaussian_noise,
    "time_stretch": time_stretch,
    "pitch_shift": pitch_shift,
    "gain": gain,
    "loudness_normalization": loudness_normalization,
    "high_pass_filter": high_pass_filter,
    "low_pass_filter": low_pass_filter,
    "trim": trim,
    "resample": resample,
    "room_simulator": room_simulator,
    "apply_impulse_response": apply_impulse_response
}

def apply_augmentation(input_file, method):
    directory = os.path.dirname(input_file)
    base_name = os.path.basename(input_file).split('.')[0]
    output_file = f"{directory}/{base_name}_{method}.wav"
    
    # Dynamically call the augmentation method
    augment_func = augmentations.get(method)
    if augment_func:
        augment_func(input_file, output_file)
        print(f"Applied {method} and saved to {output_file}")
    else:
        print(f"Method {method} not found!")

input_file = "Project/assets/335/335_AUDIO-1.wav"
method = "pitch_shift"
apply_augmentation(input_file, method)


# hilangin legend di plot_waveform_and_mel_spectrogram