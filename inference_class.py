import numpy as np
import sounddevice as sd
from joblib import load
from classify_utilities import AudioProcessor
from pyartnet import ArtNetNode
import asyncio

# import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite

# import tensorflow as tf


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
            fmax=config["fmax"],
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            data_range=config["data_range"],
        )

        self.artnet_node = None
        self.last_prediction = None

        try:
            # model = tf.keras.load_model(config["model_path"])
            self.interpreter = tflite.Interpreter(model_path=config["model_path"])
            self.interpreter.allocate_tensors()
            # Get input and output tensors
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.labels_encoder = load(config["labels_path"])
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

    async def listen_and_predict(self, duration=1.0, overlap=0.5):
        """Listen to live audio and make predictions."""
        sample_rate = self.config["sample_rate"]
        buffer_length = int(sample_rate * duration)
        hop_length = int(sample_rate * overlap)
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
                        prediction_feature = self.audio_processor(
                            data=buffer, data_range=self.config["data_range"]
                        )
                        reshaped_feature = prediction_feature.reshape(
                            1,
                            self.config["mel_frames"],
                            self.config["num_mels"],
                            self.config["num_channels"],
                        )

                        # prediction = model.predict(reshaped_feature)
                        self.interpreter.set_tensor(
                            self.input_details[0]["index"], reshaped_feature
                        )
                        self.interpreter.invoke()
                        prediction = self.interpreter.get_tensor(
                            self.output_details[0]["index"]
                        )

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

                        # Shift the buffer by 'hop_length' and read the next chunk
                        next_chunk, _ = stream.read(hop_length)
                        buffer = np.roll(buffer, -hop_length)
                        buffer[-hop_length:] = next_chunk.flatten()

                except Exception as e:
                    print(f"Error while recording: {e}")
                    break

    def idx2label(self, idx, encoder):
        idx_reshaped = np.array(idx).reshape(1, -1)
        return encoder.inverse_transform(idx_reshaped)[0][0]
