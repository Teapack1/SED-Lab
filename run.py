import os
import argparse
from inference_class import SoundClassificationService
import asyncio
import json

# Paths to model and labeler
DATA_DIR = "data"
MODEL_PATH = os.path.join("model", "model.keras")
LABELER_PATH = os.path.join("model", "label_encoder.joblib")
CONFIG_PATH = os.path.join("model", "config.json")

# Recording parameters
AUDIO_CHUNK = 0.4  # seconds
CHUNK_OVERLAP = 0.4  # seconds
NUM_CHANNELS = 1
SAMPLE_RATE = 44100
DATA_RANGE = 255  # 1 or 255

# Extracting features
MAIN_FEATURE = "mel"
N_MELS = 256
NFFT = 2048
N_MFCC = 40
FMAX = SAMPLE_RATE // 2
HOP_LENGTH = 256
N_FRAMES = round(SAMPLE_RATE * AUDIO_CHUNK / HOP_LENGTH)

# Artnet config
ARTNET_IP = "127.0.0.1"
ARTNET_UNIVERSE = 0
ARTNET_CHANNEL = 0

# ***  Input shape *** #
if MAIN_FEATURE == "mfcc":
    INPUT_SHAPE = (N_FRAMES, N_MFCC, NUM_CHANNELS)
elif MAIN_FEATURE == "mel":
    INPUT_SHAPE = (N_FRAMES, N_MELS, NUM_CHANNELS)
elif MAIN_FEATURE == "stft":
    INPUT_SHAPE = (N_FRAMES, NFFT // 2 + 1, NUM_CHANNELS)

try:
    with open(CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print("Config file not found. Using default values.")


async def main_async():
    parser = argparse.ArgumentParser(description="Audio Classification Service")
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--labeler_path",
        type=str,
        default=LABELER_PATH,
        help="Path to labeler file",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=SAMPLE_RATE,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=NUM_CHANNELS,
        help="Number of audio channels",
    )
    parser.add_argument(
        "--audio_chunk",
        type=float,
        default=AUDIO_CHUNK,
        help="Length of audio slice in seconds",
    )
    parser.add_argument(
        "--num_mels",
        type=int,
        default=N_MELS,
        help="Number of Mel bands to generate",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=NFFT,
        help="Number of samples in each FFT window",
    )
    parser.add_argument(
        "--fmax",
        type=int,
        default=FMAX,
        help="Maximum frequency when computing MEL spectrograms",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=HOP_LENGTH,
        help="Number of samples between successive FFT windows",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=N_FRAMES,
        help="Number of Mel frames",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for making predictions",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=float,
        default=CHUNK_OVERLAP,
        help="Hop length for listening in seconds",
    )
    parser.add_argument(
        "--trigger_words",
        nargs="+",
        default=["cheers2"],  # Default trigger words
        help="List of words that will trigger the action",
    )
    parser.add_argument(
        "--artnet_channel",
        type=int,
        default=ARTNET_CHANNEL,
        help="Art-Net channel to send data",
    )
    parser.add_argument(
        "--artnet_universe",
        type=int,
        default=ARTNET_UNIVERSE,
        help="Art-Net universe to send data",
    )
    parser.add_argument(
        "--artnet_ip",
        type=str,
        default=ARTNET_IP,
        help="IP address of the Art-Net node",
    )
    parser.add_argument(
        "--data_range",
        type=int,
        default=DATA_RANGE,
        help="Range of data values (1 or 255)",
    )
    parser.add_argument(
        "--n_mfcc",
        type=int,
        default=N_MFCC,
        help="Number of MFCCs to extract",
    )

    args = parser.parse_args()

    config = {
        # Paths
        "model_path": args.model_path,
        "labels_path": args.labeler_path,
        # Recording parameters
        "audio_chunk": args.audio_chunk,
        "data_range": args.data_range,
        "num_channels": args.num_channels,
        "sample_rate": args.sample_rate,
        "overlap": args.chunk_overlap,
        # Extracting features
        "num_mels": args.num_mels,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_frames": args.n_frames,
        "n_mfcc": args.n_mfcc,
        "fmax": args.fmax,
        # Prediction parameters
        "confidence_threshold": args.confidence_threshold,
        "device": "cpu",
        "trigger_words": args.trigger_words,
        "input_shape": INPUT_SHAPE,
        # Artnet config
        "artnet_channel": args.artnet_channel,
        "artnet_universe": args.artnet_universe,
        "artnet_ip": args.artnet_ip,
    }

    service = SoundClassificationService.get_instance(config)
    await service.async_init()
    await service.listen_and_predict()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


#########################
"""
import os
import argparse
from inference_class import SoundClassificationService
import asyncio
import json

# Paths to model and labeler
DATA_DIR = "data"
MODEL_PATH = os.path.join("model", "model.keras")
LABELER_PATH = os.path.join("model", "label_encoder.joblib")
CONFIG_PATH = os.path.join("model", "config.json")

# Default recording parameters
AUDIO_CHUNK = 0.4  # seconds
CHUNK_OVERLAP = 0.4  # seconds
NUM_CHANNELS = 1
SAMPLE_RATE = 44100
DATA_RANGE = 255  # 1 or 255

# Default extracting features parameters
MAIN_FEATURE = "mel"
N_MELS = 256
NFFT = 2048
N_MFCC = 40
FMAX = SAMPLE_RATE // 2
HOP_LENGTH = 256
N_FRAMES = round(SAMPLE_RATE * AUDIO_CHUNK / HOP_LENGTH)

# Default Artnet config
ARTNET_IP = "127.0.0.1"
ARTNET_UNIVERSE = 0
ARTNET_CHANNEL = 0

# ***  Input shape *** #
if MAIN_FEATURE == "mfcc":
    INPUT_SHAPE = (N_FRAMES, N_MFCC, NUM_CHANNELS)
elif MAIN_FEATURE == "mel":
    INPUT_SHAPE = (N_FRAMES, N_MELS, NUM_CHANNELS)
elif MAIN_FEATURE == "stft":
    INPUT_SHAPE = (N_FRAMES, NFFT // 2 + 1, NUM_CHANNELS)

# Load configuration from JSON file if it exists
def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, "r") as config_file:
            return json.load(config_file)
    return {}

async def main_async():
    parser = argparse.ArgumentParser(description="Audio Classification Service")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the trained model file")
    parser.add_argument("--labeler_path", type=str, default=LABELER_PATH, help="Path to labeler file")
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE, help="Audio sample rate")
    parser.add_argument("--num_channels", type=int, default=NUM_CHANNELS, help="Number of audio channels")
    parser.add_argument("--audio_chunk", type=float, default=AUDIO_CHUNK, help="Length of audio slice in seconds")
    parser.add_argument("--num_mels", type=int, default=N_MELS, help="Number of Mel bands to generate")
    parser.add_argument("--n_fft", type=int, default=NFFT, help="Number of samples in each FFT window")
    parser.add_argument("--fmax", type=int, default=FMAX, help="Maximum frequency when computing MEL spectrograms")
    parser.add_argument("--hop_length", type=int, default=HOP_LENGTH, help="Number of samples between successive FFT windows")
    parser.add_argument("--n_frames", type=int, default=N_FRAMES, help="Number of Mel frames")
    parser.add_argument("--confidence_threshold", type=float, default=0.6, help="Confidence threshold for making predictions")
    parser.add_argument("--chunk_overlap", type=float, default=CHUNK_OVERLAP, help="Hop length for listening in seconds")
    parser.add_argument("--trigger_words", nargs="+", default=["cheers2"], help="List of words that will trigger the action")
    parser.add_argument("--artnet_channel", type=int, default=ARTNET_CHANNEL, help="Art-Net channel to send data")
    parser.add_argument("--artnet_universe", type=int, default=ARTNET_UNIVERSE, help="Art-Net universe to send data")
    parser.add_argument("--artnet_ip", type=str, default=ARTNET_IP, help="IP address of the Art-Net node")
    parser.add_argument("--data_range", type=int, default=DATA_RANGE, help="Range of data values (1 or 255)")
    parser.add_argument("--n_mfcc", type=int, default=N_MFCC, help="Number of MFCCs to extract")

    args = parser.parse_args()

    # Load configuration from file if it exists
    config = load_config(CONFIG_PATH)

    # Merge configuration with command-line arguments, giving precedence to the config file
    final_config = {
        # Paths
        "model_path": MODEL_PATH,
        "labels_path": LABELER_PATH,
        # Recording parameters
        "audio_chunk": config.get("audio_chunk", AUDIO_CHUNK),
        "data_range": config.get("data_range", DATA_RANGE),
        "num_channels": config.get("num_channels", NUM_CHANNELS),
        "sample_rate": config.get("sample_rate", SAMPLE_RATE),
        # Extracting features
        "num_mels": config.get("num_mels", N_MELS),
        "n_fft": config.get("n_fft", NFFT),
        "hop_length": config.get("hop_length", HOP_LENGTH),
        "n_frames": config.get("n_frames", N_FRAMES),
        "n_mfcc": config.get("n_mfcc", N_MFCC),
        "fmax": config.get("fmax", FMAX),
        # Prediction parameters
        "confidence_threshold": args.confidence_threshold,
        "device": "cpu",
        "trigger_words": args.trigger_words,
        "input_shape": INPUT_SHAPE,
        "overlap": args.chunk_overlap,
        # Artnet config
        "artnet_channel": args.artnet_channel,
        "artnet_universe": args.artnet_universe,
        "artnet_ip": args.artnet_ip,
    }

    service = SoundClassificationService.get_instance(final_config)
    await service.async_init()
    await service.listen_and_predict()

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()

"""
