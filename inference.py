# main.py

import os
import argparse
from src.inference_class import SoundClassificationService
import asyncio
import json

CONFIG_PATH = "config.json"


async def main_async():
    parser = argparse.ArgumentParser(description="Audio Classification Service")
    parser.add_argument(
        "-conf",
        "--config_path",
        type=str,
        default=CONFIG_PATH,
        help="Path to the trained model file",
    )

    parser.add_argument(
        "-f",
        "--feature",
        type=str,
        default="mel",  # mel, mfcc, stft
        help="What feature type does the model expect? (mel, mfcc, stft)",
    )

    parser.add_argument(
        "--scaler_path",
        type=str,
        help="Path to the feature scaler file",
    )

    parser.add_argument(
        "--lite_model_path",
        type=str,
        help="Path to the trained lite model file",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--labeler_path",
        type=str,
        help="Path to labeler file",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=1,
        help="Number of audio channels",
    )
    parser.add_argument(
        "--audio_chunk",
        type=float,
        default=0.4,
        help="Length of audio slice in seconds",
    )
    parser.add_argument(
        "--num_mels",
        type=int,
        default=256,
        help="Number of Mel bands to generate",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=2048,
        help="Number of samples in each FFT window",
    )
    parser.add_argument(
        "--fmax",
        type=int,
        default=22050,
        help="Maximum frequency when computing MEL spectrograms",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=512,
        help="Number of samples between successive FFT windows",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=34,
        help="Number of frames of audio to use for prediction",
    )
    parser.add_argument(
        "-prob",
        "--probability_threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "-o",
        "--chunk_overlap",
        type=float,
        default=0.2,
        help="Recording sample overlap in seconds",
    )
    parser.add_argument(
        "-tw",
        "--trigger_words",
        nargs="+",
        default=["cheers, silence"],  # Default trigger words
        help="List of words that will trigger the action",
    )
    parser.add_argument(
        "-ac",
        "--artnet_channel",
        type=int,
        default=0,
        help="Art-Net channel to send data",
    )
    parser.add_argument(
        "-au",
        "--artnet_universe",
        type=int,
        default=0,
        help="Art-Net universe to send data",
    )
    parser.add_argument(
        "-aip",
        "--artnet_ip",
        type=str,
        default="127.0.0.1",
        help="IP address of the Art-Net node",
    )
    parser.add_argument(
        "--data_range",
        type=int,
        default=255,
        help="Range of data values (1 or 255)",
    )
    parser.add_argument(
        "--n_mfcc",
        type=int,
        default=40,
        help="Number of MFCCs to extract",
    )

    parser.add_argument(
        "-dev",
        "--device",
        type=str,
        default="cpu",
        help="processing unit",
    )

    parser.add_argument(
        "-mf",
        "--model_format",
        type=str,
        default="keras",
        help="model format",
    )

    parser.add_argument(
        "-mic",
        "--mic_device",
        type=int,
        default=0,
        help="microphone device index",
    )

    args = parser.parse_args()

    try:
        with open(args.config_path, "r") as config_file:
            config = json.load(config_file)
            print("\nConfig file loaded succesfuly.\n")
    except FileNotFoundError:
        print("\nConfig file not found. Using default values.\n")

    inference_parameters = {
        # Paths
        "model_path": config.get("model_path", args.model_path),
        "labeler_path": config.get("labeler_path", args.labeler_path),
        "lite_model_path": config.get("lite_model_path", args.lite_model_path),
        "model_format": config.get("model_format", args.model_format),
        "scaler_path": config.get("scaler_path", args.scaler_path),
        "scaler_type": config.get("scaler_type", "fitted"),
        # Recording parameters
        "audio_chunk": config.get("audio_chunk", args.audio_chunk),
        "data_range": config.get("data_range", args.data_range),
        "num_channels": config.get("num_channels", args.num_channels),
        "sample_rate": config.get("sample_rate", args.sample_rate),
        "overlap": args.chunk_overlap,
        # Extracting features
        "main_feature": config.get("main_feature", args.feature),
        "num_mels": config.get("num_mels", args.num_mels),
        "n_fft": config.get("n_fft", args.n_fft),
        "hop_length": config.get("hop_length", args.hop_length),
        "n_frames": config.get("n_frames", args.n_frames),
        "n_mfcc": config.get("n_mfcc", args.n_mfcc),
        "fmax": config.get("fmax", args.fmax),
        # Prediction parameters
        "probability_threshold": args.probability_threshold,
        "device": args.device,
        "mic_device": args.mic_device,
        "trigger_words": args.trigger_words or config.get("labels"),
        # Artnet config
        "artnet_channel": args.artnet_channel,
        "artnet_universe": args.artnet_universe,
        "artnet_ip": args.artnet_ip,
    }

    service = SoundClassificationService.get_instance(inference_parameters)
    await service.async_init()
    await service.listen_and_predict()


def main():
    asyncio.run(main_async())


# python inference.py -conf MODEL/config.json -prob 0.9 -tw "cheers, silence" -o 0.1 -ac 0 -au 0 -aip "127.0.0.1"

if __name__ == "__main__":
    main()
