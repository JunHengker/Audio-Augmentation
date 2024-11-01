import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import soundfile as sf
import numpy as np
from audiomentations import (PitchShift, Shift, AddBackgroundNoise, AddGaussianSNR)

def load_audio(input_file):
    return librosa.load(input_file, sr=None)

def save_audio(output_file, y, sr):
    sf.write(output_file, y, sr)

def save_mel_spectrogram(audio, sr, output_image_path):
    plt.figure(figsize=(6, 4))
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, fmax=8000)
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()

def time_shift(input_file, output_file):
    y, sr = load_audio(input_file)
    augment = Shift(min_shift=-0.1, max_shift=0.1, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    save_audio(output_file, y_augmented, sr)
    return y, y_augmented, sr

def pitch_shift(input_file, output_file):
    y, sr = load_audio(input_file)
    augment = PitchShift(min_semitones=-1.5, max_semitones=-1.5, p=1.0)  # ubah jadi -1.5
    y_augmented = augment(samples=y, sample_rate=sr)
    save_audio(output_file, y_augmented, sr)
    return y, y_augmented, sr

def add_background_noise(input_file, output_file):
    y, sr = load_audio(input_file)
    augment = AddBackgroundNoise(sounds_path="Project/assets/background_noises", min_snr_in_db=0, max_snr_in_db=10, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    save_audio(output_file, y_augmented, sr)
    return y, y_augmented, sr

def add_gaussian_snr(input_file, output_file):
    y, sr = load_audio(input_file)
    augment = AddGaussianSNR(min_snr_in_db=25, max_snr_in_db=30, p=1.0)
    y_augmented = augment(samples=y, sample_rate=sr)
    save_audio(output_file, y_augmented, sr)
    return y, y_augmented, sr


augmentations = {
    "time_shift": time_shift,
    "pitch_shift": pitch_shift,
    "add_background_noise": add_background_noise,
    "add_gaussian_snr": add_gaussian_snr
}

def process_files(directory, method):
    augment_func = augmentations.get(method)
    if augment_func:
        for filename in os.listdir(directory):
            input_file = os.path.join(directory, filename)
                
            # Output files
            base_name = os.path.splitext(filename)[0]
            output_file = f"Project/assets/Train_Datasets/Noise_Injection/{base_name}_{method}.wav" 
            # the augmented audio ubah (output audio)
            output_image_after = f"Project/assets/Train_Spectograms/Noise_Injection/{base_name}_{method}.png" 
            # augmented specto ubah (output image)
               
            #process the audio
            y, y_augmented, sr = augment_func(input_file, output_file)
            save_mel_spectrogram(y_augmented, sr, output_image_after)
                
            print(f"Processed: {filename}, Saved as: {output_file}")
    else:
        print(f"Method {method} not found!")

def process_files_with_prefix(directory, prefix, method):
    augment_func = augmentations.get(method)
    if augment_func:
        for filename in os.listdir(directory):
            if filename.startswith(prefix):
                input_file = os.path.join(directory, filename)
                
                # Output files
                base_name = os.path.splitext(filename)[0]
                output_file = f"Project/assets/Train_Datasets/Time_Shift/{base_name}_{method}.wav" 
                # the augmented audio ubah (output audio)
                output_image_after = f"Project/assets/Train_Spectograms/Time_Shift/{base_name}_{method}.png" 
                # augmented specto ubah (output image)
               
                #process the audio
                y, y_augmented, sr = augment_func(input_file, output_file)
                save_mel_spectrogram(y_augmented, sr, output_image_after)
                
                print(f"Processed: {filename}, Saved as: {output_file}")
    else:
        print(f"Method {method} not found!")

# Directory and parameters
directory = "Project/assets/Before_Test_54_Sample/noise_injection" # ubah (direktori input)
prefix = "" # ubah (prefix angka, jadi main per batch)
method = "add-gaussian_snr" # ubah (metodeny sesuai yang ud di define diatas)

# process_files_with_prefix(directory, prefix, method)
# process_files(directory, method)
