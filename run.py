import os
import argparse
from inference_class import SoundClassificationService
import asyncio

# Paths to model and labeler
DATA_DIR = "data"
MODEL_PATH = os.path.join("model", "model.keras")
LABELER_PATH = os.path.join("model", "label_encoder.joblib")

# Recording parameters
AUDIO_CHUNK = 0.4  # seconds
LISTENING_HOP_LENGTH = 0.4  # seconds
NUM_CHANNELS = 1
SAMPLE_RATE = 48000

# Extracting features
DATA_RANGE = 255  # 1 or 255
MEL_FRAMES = 35
N_MELS = 256
NFFT = 2048
FMAX = SAMPLE_RATE // 2
HOP_LENGTH = 512

# Artnet config
ARTNET_IP = "127.0.0.1"
ARTNET_UNIVERSE = 0
ARTNET_CHANNEL = 0


async def main_async():
    parser = argparse.ArgumentParser(description="Audio Classification Service")
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--labeler_path", type=str, default=LABELER_PATH, help="Path to labeler file"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=SAMPLE_RATE, help="Audio sample rate"
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
        "--num_mels", type=int, default=N_MELS, help="Number of Mel bands to generate"
    )
    parser.add_argument(
        "--n_fft", type=int, default=NFFT, help="Number of samples in each FFT window"
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
        "--mel_frames",
        type=int,
        default=MEL_FRAMES,
        help="Number of samples between successive FFT windows",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for making predictions",
    )
    parser.add_argument(
        "--listening_hop_length",
        type=float,
        default=LISTENING_HOP_LENGTH,
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
        help="Number of samples between successive FFT windows",
    )
    parser.add_argument(
        "--artnet_universe",
        type=int,
        default=ARTNET_UNIVERSE,
        help="Number of samples between successive FFT windows",
    )
    parser.add_argument(
        "--artnet_ip",
        type=str,
        default=ARTNET_IP,
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--data_range",
        type=int,
        default=DATA_RANGE,
        help="Number of samples between successive FFT windows",
    )

    args = parser.parse_args()

    config = {
        "model_path": args.model_path,
        "labels_path": args.labeler_path,
        "sample_rate": args.sample_rate,
        "num_channels": args.num_channels,
        "audio_chunk": args.audio_chunk,
        "mel_frames": args.mel_frames,
        "num_mels": args.num_mels,
        "n_fft": args.n_fft,
        "fmax": args.fmax,
        "hop_length": args.hop_length,
        "confidence_threshold": args.confidence_threshold,
        "device": "cpu",
        "trigger_words": args.trigger_words,
        "artnet_channel": args.artnet_channel,
        "artnet_universe": args.artnet_universe,
        "artnet_ip": args.artnet_ip,
        "data_range": args.data_range,
    }

    service = SoundClassificationService.get_instance(config)
    await service.async_init()
    await service.listen_and_predict()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
