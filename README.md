# Audio Augmentation Libraries

In this project, several Python libraries were used for audio processing and augmentation.

## 1. **Librosa**

Librosa is a powerful library for audio and music analysis. It is primarily used for loading and processing audio files, especially `.wav` format. Some of the key features provided by Librosa include:

- **Loading Audio Files**: Librosa is used to load audio files into memory for further processing. It can handle `.wav`, `.mp3`, and other common formats.
- **Basic Audio Augmentation**: While Librosa is not specifically designed for augmentation, it can perform light modifications such as:
  - Time-stretching the audio.
  - Pitch shifting.
  - Adding noise.

Librosa also provides advanced functionality like spectrogram generation and audio feature extraction, making it a versatile tool for audio preprocessing.

## 2. **Soundfile**

Soundfile is a simple and efficient library for reading and writing sound files. It supports a wide range of audio formats, including `.wav`, `.flac`, and `.ogg`. Key uses in this project include:

- **Reading Audio Files**: Soundfile is used to read and write audio data from and to disk.
- **File I/O Operations**: This library enables easy manipulation of audio file formats and ensures compatibility with other audio libraries like Librosa.

Soundfile operates efficiently with NumPy arrays, making it a natural fit for projects that need to handle large amounts of audio data.

## 3. **audiomentations**

Audiomentations is a specialized library for audio data augmentation. It provides several augmentation techniques that are directly applicable to machine learning models. The main reason audiomentations is used in this project is its **compatibility with machine learning (ML)** tasks. It integrates smoothly into ML pipelines, making it an ideal choice for augmenting audio data in preparation for model training.

Some of the augmentations available in audiomentations include:

- Adding background noise.
- Time-stretching and pitch-shifting.
- Random volume changes.
- Shifting the audio forwards or backwards in time.
- Chain Augmentation.

These augmentations are critical for increasing the diversity of the training data, thus helping to improve the robustness of machine learning models. Audiomentations is preferred in this project due to its focus on audio augmentation for ML purposes, ensuring that the transformations it applies are appropriate for training deep learning models.

---

# Requirements

## 1. **Install Librosa**

pip install librosa

## 2. **Install Audiomentations**

pip install audiomentations
