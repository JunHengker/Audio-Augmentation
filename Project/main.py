import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os


# Total 41 augmentations available in audiomentations library
from audiomentations import (PitchShift, Shift, TimeMask)

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

# 1. Time Shift
def time_shift(input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    # Tune here
    augment = Shift(min_shift=-0.1, max_shift=0.1, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="Time Shift")
    
    # plt.show()

# 2. Pitch Shift
def pitch_shift(input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    # Tune here
    augment = PitchShift(min_semitones=-1, max_semitones=1, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="Pitch Shift")
    
    # plt.show()

# 3. Time Masking
def time_mask(input_file, output_file):
    y, sr = load_audio(input_file)
    plot_waveform_and_mel_spectrogram(y, sr, title="Original Audio")
    
    # Tune here
    augment = TimeMask(min_band_part=0.05, max_band_part=0.1, fade=True, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    
    save_audio(output_file, y_augmented, sr)
    plot_waveform_and_mel_spectrogram(y_augmented, sr, title="Time Masking")
    
    # plt.show()

# Mapping method names to function calls
augmentations = {
    "time_shift": time_shift,
    "pitch_shift": pitch_shift,
    "time_mask": time_mask
}

# Function to apply augmentation (Testing on a single file)
def apply_augmentation(input_file, method):
    directory = os.path.dirname(input_file)
    base_name = os.path.basename(input_file).split('.')[0]
    output_file = f"Project/assets/Train_Datasets/{base_name}_{method}.wav"
    
    # Dynamically call the augmentation method
    augment_func = augmentations.get(method)
    if augment_func:
        augment_func(input_file, output_file)
        print(f"Applied {method} and saved to {output_file}")
    else:
        print(f"Method {method} not found!")

# Testing on a single file, disable when applying to multiple files
# input_file = "Project/assets/etc/335/335_AUDIO-1.wav"
# method = "time_shift"
# apply_augmentation(input_file, method)

# Mass apply augmentations
def process_files_with_prefix(directory, prefix, method):
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            input_file = os.path.join(directory, filename)
            apply_augmentation(input_file, method) 
            # to change the output directory, change the output_file in apply_augmentation


directory = "Project/assets/Before_Train_270_Sample"
prefix = "414"
method = "time_shift"

# Remember do disable the plot function, or ur laptop will be
# BLOWWNNN jun
process_files_with_prefix(directory, prefix, method)