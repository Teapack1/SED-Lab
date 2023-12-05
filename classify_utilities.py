import struct
import librosa
import numpy as np
import pandas as pd


class AudioProcessor:
    def __init__(
        self,
        sample_rate=22050,
        n_mels=128,
        fmax=11000,
        n_fft=2048,
        hop_length=512,
        audio_chunk=0.5,
        slice_audio=False,
        data_range=1.0,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fmax = fmax
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.audio_chunk = audio_chunk
        self.slice_audio = slice_audio
        self.data_range = data_range

    def __call__(self, data, data_range):
        self.data_range = data_range
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

        spectrogram = librosa.feature.melspectrogram(
            y=data,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            fmax=self.fmax,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # general scale normalization with min-max:
        # min_db, max_db = -60, 80
        # spectrogram = np.clip((spectrogram - min_db) / (max_db - min_db), 0, 1)
        # spectrogram *= 255

        # Min-Max normalization
        spectrogram = (spectrogram - spectrogram.min()) / (
            spectrogram.max() - spectrogram.min()
        )

        # Scale the data range
        spectrogram *= self.data_range

        return spectrogram.T

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
