# Audio Classify 2
Audio-classification task with CNNs and Tensorflow 2 framework. Training notebook with exploratory data analysis and features extraction and model class to test several model architectures. Automatic testing on whole new data. Real-time inference using audio device, sends triggers to specified Artnet universe.
## Key Features
- Real-time audio classification.
- Customizable model settings for different use cases.
- Extensive audio and label processing functions.
- Notebook-style training script for easy model training.
- Comprehensive inference class for real-time applications.
- Trigger-based system for specific audio patterns.

## Installation
```
bash
git clone https://github.com/your-repository/AudioClassify2.git
cd AudioClassify2
pip install -r requirements.txt
```

## Usage and Examples
### Basic Usage
To start using Audio Classify 2, simply import the necessary classes and initiate the audio classification service:
```
from inference_class import SoundClassificationService

config = {
    # Configuration settings
}
service = SoundClassificationService.get_instance(config)
service.async_init()
service.listen_and_predict()
```
### Advanced Usage
For more advanced use cases, you can customize the configuration settings:
```
config = {
    "model_path": "path/to/model.keras",
    "labels_path": "path/to/label_encoder.joblib",
    "sample_rate": 44100,
    # Additional settings...
}
```

## File Breakdown
- `training_script.txt`: Notebook-style script for training the model.
- `classify_utilities.py`: Contains functions for audio processing and label handling.
- `inference_class.py`: Defines the real-time inference class.
- `model.py`: Includes various model architectures.
- `run.py`: Executes real-time inference and triggers actions based on audio input.

## Customization and Settings
You can customize various parameters in `run.py`:
- `AUDIO_CHUNK`: Length of audio slice in seconds.
- `SAMPLE_RATE`: Audio sample rate.
- `N_MELS`: Number of Mel bands.
- `MODEL_PATH`: Path to the trained model.

### Recommended Settings
- **Home Automation:** `SAMPLE_RATE`: 44100, `N_MELS`: 40
- **Security Monitoring:** `AUDIO_CHUNK`: 0.5, `SAMPLE_RATE`: 48000

## Last Remarks and Tips
- Ensure your audio input device is properly configured.
- Experiment with different `AUDIO_CHUNK` values for optimal performance.
- Check the `training_script.txt` for model training guidelines.

## FAQ
- **Q: Can I use my own audio model?**
  - A: Yes, you can train and integrate your model using the training script.
- **Q: Is it possible to run this on a Raspberry Pi?**
  - A: Yes, but ensure you have the necessary audio input hardware.

## License
This project is licensed under the Apache 2 License - see the [LICENSE.md](LICENSE) file for details.

---



