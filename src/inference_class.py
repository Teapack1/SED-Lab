# src/inference_class.py

import numpy as np
import sounddevice as sd
from joblib import load
from src.audio_processor import AudioProcessor
from pyartnet import ArtNetNode
import asyncio
import joblib
import traceback


# import tflite_runtime.interpreter as tflite
# import tensorflow.lite as tflite

import tensorflow as tf


class SoundClassificationService:
    _instance = None

    def __init__(self, config):
        """Initialize the service with the given configuration."""
        self.config = config

        self.microphone_index = sd.query_devices(kind="input")["index"]
        print(self.microphone_index)
        self.audio_processor = AudioProcessor(
            sample_rate=config["sample_rate"],
            n_mels=config["num_mels"],
            n_mfcc=config["n_mfcc"],
            fmax=config["fmax"],
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            audio_chunk=config["audio_chunk"],
            data_range=config["data_range"],
            main_feature=config["main_feature"],
            scaler_type=config["scaler_type"],
            scaler_path=config["scaler_path"],
            use_delta=False,
            slice_audio=False,
        )

        self.artnet_node = None
        self.last_prediction = None
        # ***  Input shape *** #
        if config["main_feature"] == "mfcc":
            self.input_shape = (
                config["n_frames"],
                config["n_mfcc"],
                config["num_channels"],
            )
        elif config["main_feature"] == "mel":
            self.input_shape = (
                config["n_frames"],
                config["num_mels"],
                config["num_channels"],
            )
        elif config["main_feature"] == "stft":
            self.input_shape = (
                config["n_frames"],
                config["n_fft"] // 2 + 1,
                config["num_channels"],
            )

        try:
            self.labels_encoder = joblib.load(config["labeler_path"])
            self.scaler = joblib.load(config["scaler_path"])
            try:
                self.model = tf.keras.models.load_model(
                    f"{config['model_path']}.{config['model_format']}"
                )
            # file not found
            except FileNotFoundError as e:
                self.model = tf.keras.models.load_model(f"{config['model_path']}")

            except Exception as e:
                print(f"Error loading main model: {e}")
                raise

            """
            self.interpreter = tflite.Interpreter(model_path=config["model_path"])
            self.interpreter.allocate_tensors()
            # Get input and output tensors
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            """

        except Exception as e:
            print(f"Error loading files: {e}")
            raise

    async def async_init(self):
        self.artnet_node = ArtNetNode(ip=self.config["artnet_ip"], port=6454)
        self.artnet_universe = self.artnet_node.add_universe(
            self.config["artnet_universe"]
        )
        self.artnet_channel = self.artnet_universe.add_channel(start=1, width=1)
        self.last_trigger_time = 0
        self.trigger_cooldown = 5  # seconds

    @classmethod
    def get_instance(cls, config):
        """Asynchronously get the instance of the class."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    async def listen_and_predict(self):
        """Listen to live audio and make predictions."""
        sample_rate = self.config["sample_rate"]
        buffer_length = int(sample_rate * self.config["audio_chunk"])
        overlap = int(sample_rate * self.config["overlap"])
        buffer = np.zeros(buffer_length)

        with sd.InputStream(
            samplerate=sample_rate,
            device=self.microphone_index,
            channels=self.config["num_channels"],
        ) as stream:
            print("Listening... Press Ctrl+C to stop.")
            while True:
                try:
                    # Read the first chunk to fill the buffer
                    audio_chunk, _ = stream.read(buffer_length)
                    buffer = audio_chunk.flatten()

                    while True:
                        # Process the buffer
                        prediction_feature = self.audio_processor(data=buffer)

                        reshaped_feature = prediction_feature.reshape(
                            1,  # batch size
                            self.input_shape[0],  # Frames
                            self.input_shape[1],  # freq bands
                            self.input_shape[2],  # channels
                        )

                        prediction = self.model.predict(reshaped_feature)

                        """
                        self.interpreter.set_tensor(
                            self.input_details[0]["index"], reshaped_feature
                        )
                        self.interpreter.invoke()
                        prediction = self.interpreter.get_tensor(
                            self.output_details[0]["index"]
                        )
                        """

                        keyword = self.idx2label(prediction, self.labels_encoder)
                        if keyword:
                            print(
                                f"Predicted Keyword: {keyword}, with: {prediction * 100}"
                            )

                        # Artnet trigger logic
                        if (
                            keyword in self.config["trigger_words"]
                            and self.last_prediction not in self.config["trigger_words"]
                        ):
                            self.artnet_channel.add_fade([1], 0)
                            await self.artnet_channel
                            await asyncio.sleep(
                                0.1
                            )  # Short delay for the remote device to recognize '1'
                            self.artnet_channel.add_fade([0], 0)
                            await self.artnet_channel
                            self.last_prediction = keyword
                        else:
                            self.artnet_channel.add_fade([0], 0)
                            await self.artnet_channel
                            if keyword not in self.config["trigger_words"]:
                                self.last_prediction = keyword

                        # Shift the buffer by 'overlap' and read the next chunk
                        next_chunk, _ = stream.read(overlap)
                        buffer = np.roll(buffer, -overlap)
                        buffer[-overlap:] = next_chunk.flatten()

                except Exception as e:
                    print(f"Error while recording: {e}")
                    print("Traceback details:")
                    print(traceback.format_exc())
                    break

    def idx2label(self, idx, encoder):
        idx_reshaped = np.array(idx).reshape(1, -1)
        return encoder.inverse_transform(idx_reshaped)[0][0]
