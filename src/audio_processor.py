# src/audio_processor.py

import struct
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
)
from sklearn.exceptions import NotFittedError
import joblib


class AudioProcessor:
    def __init__(
        self,
        sample_rate=44100,
        n_mels=128,
        fmax=11000,
        n_mfcc=40,
        n_fft=2048,
        hop_length=512,
        audio_chunk=0.4,
        slice_audio=False,
        data_range=255,
        main_feature="mel",
        scaler_type="fitted",  # 'minmax', 'maxabs', 'robust', or None
        use_delta=False,
        scaler_path=None,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.fmax = fmax
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.audio_chunk = audio_chunk
        self.slice_audio = slice_audio
        self.data_range = data_range
        self.main_feature = main_feature

        self.use_delta = use_delta
        self.audio_length = int(sample_rate * audio_chunk)

        if scaler_type == "fitted":
            self.scaler = joblib.load(scaler_path)
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "maxabs":
            self.scaler = MaxAbsScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None

    def __call__(self, data):
        return self.feature_extractor(data)

    def feature_extractor(self, data):
        if self.slice_audio:
            print("sliced.")
            sample_length = self.audio_chunk * self.sample_rate

            librosa_audio_sliced = data[:sample_length]
            if len(data) < sample_length:
                librosa_audio_sliced = np.pad(
                    data, (0, sample_length - len(data)), constant_values=0
                )
            data = librosa_audio_sliced

        if self.main_feature == "mel":
            spectrogram = self.extract_mel_spectrogram(y=data)
        elif self.main_feature == "mfcc":
            spectrogram = self.extract_mfcc(y=data)
        elif self.main_feature == "stft":
            spectrogram = self.extract_spectrogram(y=data)
        else:
            raise ValueError("Invalid feature type")

        return spectrogram

    def read_file_properties(self, filename):
        wave_file = open(filename, "rb")

        riff = wave_file.read(12)
        fmt = wave_file.read(36)

        num_channels_string = fmt[10:12]
        num_channels = struct.unpack("<H", num_channels_string)[0]

        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I", sample_rate_string)[0]

        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H", bit_depth_string)[0]

        wave_file.close()

        # Load the audio file with librosa
        y, sr = librosa.load(filename, sr=None, mono=True)  # Load as mono

        # Compute RMS of the audio signal using librosa
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = np.mean(rms)  # Average RMS over time if needed
        # avg_rms = None
        # Compute the length of the audio sample in seconds
        length_in_seconds = len(y) / sr  # Total samples / Sample rate

        # Length in samples
        length_in_frames = len(y)

        return (
            num_channels,
            sample_rate,
            bit_depth,
            avg_rms,
            length_in_seconds,
            length_in_frames,
        )  # Added length_in_samples

    def envelope(self, y, rate, threshold):
        mask = []
        y = pd.Series(y).apply(np.abs)
        y_mean = y.rolling(window=int(rate / 20), min_periods=1, center=True).max()
        for mean in y_mean:
            if mean > threshold:
                mask.append(True)
            else:
                mask.append(False)
        return mask, y_mean

    def idx2label(self, idx, encoder):
        idx_reshaped = np.array(idx).reshape(1, -1)
        return encoder.inverse_transform(idx_reshaped)[0][0]

    def label2idx(label, encoder):
        label = np.array(label).reshape(-1, 1)
        return encoder.transform(label).toarray()[0]

    ############################# AUDIO FEATURES PROCESSING ##################################

    def fit_scaler(self, features):
        # Flatten features to fit the scaler on the entire dataset
        features = np.vstack(features)
        if self.scaler:
            self.scaler.fit(features)

    def post_process(self, feature):
        if self.scaler is not None:
            try:
                feature = self.scaler.transform(feature)
            except NotFittedError as e:
                print(f"Error: {e}")
                print("Scaler will be fitted later on the raw dataset...")
                return feature

        feature = (feature - feature.min()) / (feature.max() - feature.min())

        if self.data_range == 255:
            feature = feature * 255.0
            feature = feature.astype(np.uint8)
        elif self.data_range == 1:
            feature = (feature * 2) - 1

        return feature

    def extract_spectrogram(self, y):
        spectrogram = librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

        spectrogram_db = spectrogram_db.T
        spectrogram_db = self.post_process(spectrogram_db)
        return spectrogram_db  # Transpose to ensure time axis is the second dimension

    def extract_mel_spectrogram(self, y):
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        mel_spectrogram_db = mel_spectrogram_db.T
        mel_spectrogram_db = self.post_process(mel_spectrogram_db)
        return mel_spectrogram_db

    def extract_mfcc(self, y):
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        if self.use_delta:
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

        mfcc = mfcc.T
        mfcc = self.post_process(mfcc)
        return mfcc  # Transpose to ensure time axis is the second dimension

