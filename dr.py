# Audio dataset recorder and metadata creator

import sounddevice as sd
import argparse
import os
import librosa
import random
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import pandas as pd


def find_mic_index(sounddevice):
    mic_index = None
    devices = sounddevice.query_devices()

    for i, dev in enumerate(devices):
        print("Device {}: {}".format(i, dev["name"]))

        if dev["max_input_channels"] > 0:
            print("------------------------------------")
            print("Found an input: device {} - {}".format(i, dev["name"]))
            print(dev)
            mic_index = i
            return mic_index

    if mic_index is None:
        print("Using default input device.")
        return sd.default.device[0]

    return mic_index


def list_devices():
    print(sd.query_devices())


class AudioRecorder:
    def __init__(
        self,
        sample_rate=16000,
        duration=0.4,
        classes=None,
        samples_dir="audio_samples",
        aug_samples_dir="augmented_audio_samples",
        sample_count=20,
        device_index=0,
        channels=1,
        chunk_size=128,
        treshold=0.1,
    ):
        self.SAMPLE_RATE = sample_rate
        self.DURATION = duration
        self.CLASSES = classes
        self.SAMPLES_DIR = samples_dir
        self.SAMPLE_COUNT = sample_count
        self.DEVICE_INDEX = device_index
        self.CHANNELS = channels
        self.AUG_SAMPLES_DIR = aug_samples_dir
        self.CHUNK_SIZE = chunk_size
        self.TRESHOLD = treshold

        if not os.path.exists(self.SAMPLES_DIR):
            os.mkdir(self.SAMPLES_DIR)

    def apply_audio_editing(self, trim_pad_flag, normalize_flag):
        # For every directory, subdir, and file found in SAMPLES_DIR
        for dirpath, dirnames, filenames in os.walk(self.SAMPLES_DIR):
            for f in filenames:
                if f.endswith(".wav"):  # Ensure it's an audio file
                    audio_file = os.path.join(dirpath, f)

                    # Determine if we are using variant A or B
                    if os.path.dirname(audio_file) == self.SAMPLES_DIR:  # Variant A
                        edit_class_dir = (
                            self.SAMPLES_DIR
                        )  # Save directly to SAMPLES_DIR
                    else:  # Variant B
                        class_name = os.path.basename(os.path.dirname(audio_file))
                        edit_class_dir = os.path.join(
                            self.SAMPLES_DIR, class_name
                        )  # Save to the subdirectory

                    y, sr = librosa.load(audio_file, sr=None)

                    if trim_pad_flag:
                        y = self.trim_pad(y)
                        print(f"Trimmed {audio_file}")

                    if normalize_flag:
                        y = self.normalize(y)
                        print(f"Normalized {audio_file}")

                    sf.write(
                        audio_file, y, sr
                    )  # Overwrite the original file with edited data

    def trim_pad(self, audio):
        """Trim silence and pad audio to ensure consistent length."""
        # Trim silence
        trimmed, _ = librosa.effects.trim(audio, top_db=15)
        # Pad to ensure consistent length
        desired_length = int(self.SAMPLE_RATE * self.DURATION)
        if len(trimmed) < desired_length:
            padding = desired_length - len(trimmed)
            padded_audio = np.pad(trimmed, (0, padding), "constant")
        else:
            padded_audio = trimmed[:desired_length]
        return padded_audio

    def play_sample(self, filepath=None, data=None, samplerate=None):
        """Plays the audio sample at the specified file path or the provided audio data."""
        if filepath:
            # Load the audio file
            data, samplerate = sf.read(filepath)
        # Play the audio data
        sd.play(data, samplerate)
        # Use sd.wait() to block execution until audio is finished playing
        sd.wait()

    def normalize(self, audio):
        audio_max = np.max(np.abs(audio))
        if audio_max > 0:
            scaling_factor = 1.0 / audio_max
            normalized_audio = audio * scaling_factor
            return normalized_audio
        return audio.copy()

    def augment_samples(self, num_augmented, ambient_mix=False):
        class_counts = {}

        os.makedirs(self.AUG_SAMPLES_DIR, exist_ok=True)
        ambient_files = []
        if ambient_mix:
            ambient_path = os.path.join(os.getcwd(), "ambient")
            print(ambient_path)
            if os.path.exists(ambient_path):
                ambient_files = [
                    os.path.join(ambient_path, filename)
                    for filename in os.listdir(ambient_path)
                    if filename.endswith(".wav")
                ]
                print(ambient_files)

        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(self.SAMPLES_DIR)):
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                self._augment_file(
                    file_path, num_augmented, class_counts, ambient_files
                )

    def _augment_file(self, audio_file, num_augmented, class_counts, ambient_files):
        print(f"Augmenting {audio_file}")

        y, sr = librosa.load(audio_file, sr=None)

        # Determine if we are using variant A or B
        if os.path.dirname(audio_file) == self.SAMPLES_DIR:  # Variant A
            class_name = os.path.basename(audio_file).split("_")[0]
            original_prefix = class_name
            augmented_class_dir = (
                self.AUG_SAMPLES_DIR
            )  # Save directly to AUG_SAMPLES_DIR
        else:  # Variant B
            class_name = os.path.basename(os.path.dirname(audio_file))
            original_prefix = os.path.basename(audio_file).split("_")[0]
            augmented_class_dir = os.path.join(
                self.AUG_SAMPLES_DIR, class_name
            )  # Save to subdirectory
            os.makedirs(
                augmented_class_dir, exist_ok=True
            )  # Create subdirectory if it doesn't exist

        if class_name not in class_counts:
            class_counts[class_name] = 0

            # Calculate the root mean square (RMS) and db for the audio sample
        audio_rms = np.sqrt(np.mean(np.abs(y) ** 2))
        audio_db = 20 * np.log10(audio_rms) if audio_rms > 0 else -120

        for i in range(num_augmented):
            weights = [0.5, 0.5]
            mix = random.choices([True, False])
            method = random.choice(["pitch", "stretch", "noise", "db"])

            if method == "pitch":
                steps = random.randint(-3, 3)
                augmented = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

            elif method == "stretch":
                rate = random.uniform(0.8, 1.2)
                augmented = librosa.effects.time_stretch(y, rate=rate)

            elif method == "noise":
                noise_amplitude = 0.5 * audio_rms
                noise = np.random.normal(0, noise_amplitude, len(y))
                augmented = y + noise

            if method == "db":
                audio_segment = AudioSegment.from_wav(audio_file)
                # Decide on dB change based on current loudness
                if audio_db < -30:  # Very quiet
                    db_change = random.randint(2, 13)
                elif audio_db < -20:  # Quiet
                    db_change = random.randint(1, 10)
                elif audio_db < -10:  # Moderate
                    db_change = random.randint(10, 10)
                else:  # Loud
                    db_change = random.randint(-15, 2)

                augmented_segment = audio_segment.apply_gain(db_change)
                augmented = np.array(augmented_segment.get_array_of_samples())

            if ambient_files and mix:
                ambient_file = random.choice(ambient_files)
                ambient_y, _ = librosa.load(ambient_file, sr=sr)
                # Adjust length of ambient sound to match `augmented`
                if len(ambient_y) > len(augmented):
                    ambient_y = ambient_y[: len(augmented)]
                else:
                    ambient_y = np.pad(
                        ambient_y, (0, len(augmented) - len(ambient_y)), "constant"
                    )
                # Mix with a random volume ratio
                mix_ratio = random.uniform(0.1, 0.5)
                augmented = augmented * (1 - mix_ratio) + ambient_y * mix_ratio
                print(f"Mixed with {ambient_file}")

            class_counts[class_name] += 1

            new_file = os.path.join(
                augmented_class_dir,
                f"{original_prefix}_aug_{class_counts[class_name]}.wav",
            )
            sf.write(new_file, augmented, self.SAMPLE_RATE)

    def get_highest_index(self, cls, variant):
        """
        Get the highest index of the already recorded samples for a given class.
        """
        highest_index = -1

        if variant == "A":
            path = self.SAMPLES_DIR
            files = [
                f
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and f.startswith(cls)
            ]
        elif variant == "B":
            path = os.path.join(self.SAMPLES_DIR, cls)
            try:
                files = [
                    f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
                ]
            except FileNotFoundError:
                print(
                    f"Class '{cls}' not found in '{self.SAMPLES_DIR}', no samples recorded yet."
                )
                return highest_index

        for f in files:
            try:
                index = int(
                    f.split("_")[-1].split(".")[0]
                )  # Extracting index from the filename
                highest_index = max(highest_index, index)
            except ValueError:
                continue

        return highest_index

    def record_regular(
        self, record_seconds=1, channels=1, rate=16000, chunk_size=128, device=0
    ):
        recording = sd.rec(
            int(record_seconds * rate),
            samplerate=rate,
            channels=channels,
            device=device,
            dtype="int16",
        )
        sd.wait()
        return recording

    def record_auto(
        self,
        threshold=0.6,
        record_seconds=1,
        channels=1,
        rate=16000,
        chunk_size=128,
        device=0,
    ):
        def get_rms(block):
            # return np.sqrt(np.mean(np.square(block)))
            # return peak amplitude
            return np.max(np.abs(block))

        with sd.InputStream(
            channels=channels,
            samplerate=rate,
            blocksize=chunk_size,
            dtype="float32",
            device=device,
        ) as stream:
            while True:
                data, _ = stream.read(chunk_size)
                mono = data[:, 0] if channels > 1 else data
                amplitude = get_rms(mono)

                if amplitude > threshold:
                    print("* Recording with amplitude:", amplitude)
                    frames = [data]  # Start with the current chunk

                    for _ in range(1, int(rate / chunk_size * record_seconds)):
                        data, _ = stream.read(chunk_size)
                        frames.append(data)
                    return np.concatenate(frames, axis=0)

    """
    def record_auto(self, threshold=0.6, record_seconds=1, channels=1, rate=16000, chunk_size=128, overlap_factor=0.5, device=0):
        def get_rms(block):
            return np.max(np.abs(block))
        
        # Calculate the number of samples to overlap
        overlap_samples = int(chunk_size * overlap_factor)
        read_size = chunk_size - overlap_samples

        with sd.InputStream(channels=channels, samplerate=rate, blocksize=read_size, dtype='float32', device=device) as stream:
            # Initialize an empty buffer for storing overlapped chunks
            buffer = np.zeros(chunk_size, dtype='float32')
            while True:
                data, _ = stream.read(read_size)
                if channels > 1:
                    # Convert to mono by averaging the channels if stereo
                    mono = np.mean(data, axis=1)
                else:
                    mono = data[:, 0]
                
                # Shift the existing buffer and append new data for overlap
                buffer = np.roll(buffer, -read_size)
                buffer[-read_size:] = mono

                amplitude = get_rms(buffer)

                if amplitude > threshold:
                    print("* Recording with amplitude:", amplitude)
                    frames = [buffer.copy()]  # Start with the current overlapped buffer

                    for _ in range(1, int(rate / read_size * record_seconds)):
                        data, _ = stream.read(read_size)
                        if channels > 1:
                            # Convert to mono by averaging the channels if stereo
                            mono = np.mean(data, axis=1)
                        else:
                            mono = data[:, 0]
                        
                        # Update the buffer with new data
                        buffer = np.roll(buffer, -read_size)
                        buffer[-read_size:] = mono

                        frames.append(buffer.copy())
                    return np.concatenate(frames, axis=0)
    """

    def record_audio_variant_A(self, playback=False, no_listening_mode=False):
        for cls in self.CLASSES:
            highest_index = self.get_highest_index(cls, "A")
            input(f"Press Enter to start recording for class '{cls.upper()}' ...")

            for sample in range(
                highest_index + 1, highest_index + 1 + self.SAMPLE_COUNT
            ):
                print(
                    f"Recording sample '{cls}': {sample-highest_index} / {self.SAMPLE_COUNT}"
                )

                if no_listening_mode:
                    record = self.record_regular(
                        record_seconds=self.DURATION,
                        channels=self.CHANNELS,
                        rate=self.SAMPLE_RATE,
                        chunk_size=self.CHUNK_SIZE,
                        device=self.DEVICE_INDEX,
                    )
                else:
                    record = self.record_auto(
                        threshold=self.TRESHOLD,
                        record_seconds=self.DURATION,
                        channels=self.CHANNELS,
                        rate=self.SAMPLE_RATE,
                        chunk_size=self.CHUNK_SIZE,
                        device=self.DEVICE_INDEX,
                    )

                if record.shape[1] > 1:
                    record = np.mean(record, axis=1)

                filename = os.path.join(self.SAMPLES_DIR, f"{cls}_{sample}.wav")
                sf.write(filename, record, self.SAMPLE_RATE)
                print("Saved at: ", filename)

                if playback:
                    self.play_sample(data=record, samplerate=self.SAMPLE_RATE)

        print("Finished recording.")

    def record_audio_variant_B(self, playback=False, no_listening_mode=False):
        for cls in self.CLASSES:
            highest_index = self.get_highest_index(cls, "B")  # Fixed to use "B"
            input(f"Press Enter to start recording for class '{cls.upper()}' ...")

            dir = os.path.join(self.SAMPLES_DIR, cls)
            if not os.path.exists(dir):
                os.mkdir(dir)

            for sample in range(
                highest_index + 1, highest_index + self.SAMPLE_COUNT + 1
            ):
                print(
                    f"Recording sample '{cls}': {sample-highest_index} / {self.SAMPLE_COUNT}"
                )

                if no_listening_mode:
                    record = self.record_regular(
                        record_seconds=self.DURATION,
                        channels=self.CHANNELS,
                        rate=self.SAMPLE_RATE,
                        chunk_size=self.CHUNK_SIZE,
                        device=self.DEVICE_INDEX,
                    )
                else:
                    record = self.record_auto(
                        threshold=self.TRESHOLD,
                        record_seconds=self.DURATION,
                        channels=self.CHANNELS,
                        rate=self.SAMPLE_RATE,
                        chunk_size=self.CHUNK_SIZE,
                        device=self.DEVICE_INDEX,
                    )

                if record.shape[1] > 1:
                    record = np.mean(record, axis=1)

                filename = os.path.join(dir, f"{cls}_{sample}.wav")
                sf.write(filename, record, self.SAMPLE_RATE)
                print("Saved at: ", filename)

                if playback:
                    self.play_sample(data=record, samplerate=self.SAMPLE_RATE)

        print("Finished recording.")

    def produce_metadata(self):
        metadata = {"filepath": [], "label": [], "class_num": []}

        # 1) Check classes from the argument
        if self.CLASSES:
            classes = self.CLASSES
        # 2) Check AUG_SAMPLES_DIR
        elif os.path.exists(self.AUG_SAMPLES_DIR):
            direct_files = [
                f
                for f in os.listdir(self.AUG_SAMPLES_DIR)
                if os.path.isfile(os.path.join(self.AUG_SAMPLES_DIR, f))
            ]
            classes_from_files = set([f.split("_")[0] for f in direct_files])
            classes_from_dirs = set(
                [
                    d
                    for d in os.listdir(self.AUG_SAMPLES_DIR)
                    if os.path.isdir(os.path.join(self.AUG_SAMPLES_DIR, d))
                ]
            )
            classes = list(classes_from_files.union(classes_from_dirs))
        # 3) Check SAMPLES_DIR
        elif os.path.exists(self.SAMPLES_DIR):
            direct_files = [
                f
                for f in os.listdir(self.SAMPLES_DIR)
                if os.path.isfile(os.path.join(self.SAMPLES_DIR, f))
            ]
            classes_from_files = set([f.split("_")[0] for f in direct_files])
            classes_from_dirs = set(
                [
                    d
                    for d in os.listdir(self.SAMPLES_DIR)
                    if os.path.isdir(os.path.join(self.SAMPLES_DIR, d))
                ]
            )
            classes = list(classes_from_files.union(classes_from_dirs))
        # 4) Default classes
        else:
            classes = ["yes", "no", "hi"]

        # Mapping from class names to numerical values
        class_to_num = {cls: idx for idx, cls in enumerate(classes)}
        print("Class to num: ", class_to_num)

        target_dir = (
            self.AUG_SAMPLES_DIR
            if os.path.exists(self.AUG_SAMPLES_DIR)
            else self.SAMPLES_DIR
        )

        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(target_dir)):
            dir_label = os.path.basename(dirpath) if dirpath != target_dir else None

            for f in filenames:
                label = f.split("_")[0] if dir_label is None else dir_label
                file_path = os.path.join(dirpath, f)
                metadata["filepath"].append(file_path)
                metadata["label"].append(label)
                metadata["class_num"].append(
                    class_to_num.get(label, -1)
                )  # -1 as default if label is not found

        df = pd.DataFrame(metadata)

        df.to_csv("metadata.csv", index=False)


# ---------------------------- MAIN ----------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record audio samples for different classes."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["A", "B"],
        default="B",
        help="Recording method 'A' save all samples in one folder. 'B' saves samples to separate folders for each class.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augment the recorded samples with pitch, stretch, noise, and db changes. You can folder 'ambient' and put there ambient samples to mix with. (flag)",
    )
    parser.add_argument(
        "--num_augmented",
        type=int,
        default=25,
        help="Number of augmented samples for every original sample",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=False,
        default=["audio"],
        help="Specify classes for the recordings.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=20, help="Number of samples in every class."
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="Sampling Rate (16000 default)."
    )
    parser.add_argument(
        "--duration", type=float, default=1, help="Duration of one sample in seconds."
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Produce metadata after recording. (flag)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Performs peak-normalization, scales the audio so that its maximum amplitude matches a target level (1.0). (flag)",
    )
    parser.add_argument(
        "--trim_pad",
        action="store_true",
        help="Trim silence parts and pad it back with zeros to ensure consistent length. (flag)",
    )
    parser.add_argument(
        "--playback",
        action="store_true",
        help="Flag to indicate playback or specify a file for playback. (flag)",
    )
    parser.add_argument(
        "--no_listening_mode",
        action="store_true",
        help="Do not use smart recording. (flag).",
    )
    parser.add_argument(
        "-t",
        "--treshold",
        type=float,
        default=0.2,
        help="Treshold to start recording (default: 0.2).",
    )
    parser.add_argument(
        "-dev",
        "--device",
        type=int,
        default=None,
        help="Choose a specific device for recording. Lists available devices.",
    )
    args = parser.parse_args()

    if args.device:
        list_devices()
        print(
            f"Using device no. {args.device} for auto selection remove --device argument."
        )
        device_index = args.device
    else:
        device_index = find_mic_index(sd)

    recorder = AudioRecorder(
        classes=args.classes,
        sample_count=args.num_samples,
        duration=args.duration,
        device_index=device_index,
        treshold=args.treshold,
        sample_rate=args.sample_rate,
    )

    if args.method == "A":
        recorder.record_audio_variant_A(
            playback=args.playback, no_listening_mode=args.no_listening_mode
        )
    elif args.method == "B":
        recorder.record_audio_variant_B(
            playback=args.playback, no_listening_mode=args.no_listening_mode
        )
    elif args.playback not in (True, False):
        recorder.play_sample(filepath=args.playback)
        exit()

    if args.trim_pad or args.normalize:
        recorder.apply_audio_editing(args.trim_pad, args.normalize)

    if args.augment:
        recorder.augment_samples(args.num_augmented, ambient_mix=True)

    if args.metadata:
        recorder.produce_metadata()
