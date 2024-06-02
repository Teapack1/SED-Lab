# Sound Event Detection Lab

SED-Lab is universal Sound Event Detection (SED) package designed for learning and experimenting with audio classification tasks. This repository includes a training notebook and real-time inference scripts built on the TensorFlow 2 backend. It supports various audio features, multiple model architectures, and easy transfer of configuration settings via JSON files saved with trained model. 
<br>Intended usage is to experiment with niche SED tasks, use integrated <i>exploratory data analysis</i> for experimentation with different features, scalers, nn architectures and use tests to find best working parameter for specific task.
<br>The real-time inference class uses an input microphone audio device and sends triggers on the specified Artnet bus. 
<br>Use [ADRT](https://github.com/Teapack1/Audio-Dataset-Recorder-Tools) (Audio Dataset Recorder Tools) to record custom audio datasets.

![Sound Event Detection and Audio Classification](/src/img/img.jpg)

## Table of Contents
- [Sound Event Detection Lab](#sound-event-detection-lab)
  - [Table of Contents](#table-of-contents)
  - [Why Use?](#why-use)
    - [Applications and HW](#applications-and-hw)
  - [Installation](#installation)
  - [How to use?](#how-to-use)
    - [Prepare](#prepare)
    - [Training](#training)
    - [Inference](#inference)
  - [Arguments and Settings](#arguments-and-settings)
    - [Training Settings](#training-settings)
      - [Audio Samples:](#audio-samples)
      - [Audio Feature Parameters:](#audio-feature-parameters)
      - [Training Settings:](#training-settings-1)
      - [Model Settings:](#model-settings)
    - [Inference Arguments](#inference-arguments)
      - [Configuration File:](#configuration-file)
      - [Prediction and Thresholds:](#prediction-and-thresholds)
      - [Art-Net Configuration:](#art-net-configuration)
      - [Device Configuration:](#device-configuration)
    - [Parameters for Manual Setup (if config.json is not used):](#parameters-for-manual-setup-if-configjson-is-not-used)
  - [Recommended Application Settings](#recommended-application-settings)
  - [Results](#results)
  - [License](#license)


## Why Use?
- **Easy to use:** Place dataset, set up initial parameters. run training -> wait -> run inference command.
- **Educational:** The training process runs in a Jupyter notebook with detailed descriptions. Learn about audio ML, experiment with STFT, MFCCs, different normalization techniques, augmentations, and model architectures.
- **Easy deploy:** After training, a config file is generated that includes all settings from the training phase, simplifying the deployment of various models.
- **Real-time inference:** The inference script runs on real-time audio recordings and can send trigger commands via the Artnet bus.
- **Edge support:** TF-Lite models and an inference script for mobile and edge computing are available.
- **Models:** Choose from renowned model architectures, simple models, or custom-built models quickly during the training phase. Models can be browsed in the [model.md](/src/model/model.md) documentation.
 
### Applications and HW
Tested on Linux and Windows os. 
<br>Runs on cpu or CUDA gpu.
<br>Audio input device inference with overlaping samples (real-time).
<br>Offline inference on audio files (sorting).
<br>See [example applications](#recommended-application-settings) bellow.

## Installation
<b>Windows 10/11</b> setup:
1) Download and install Python 3.10.11:
<br>
https://www.python.org/downloads/release/python-31011/
<br>
`"add python.exe to PATH"` during installation.
<br>
Restart Windows

2) Clone (download) the project:
<br>
Open terminal and clone it or just download from Github.
<br>
`git clone https://github.com/Teapack1/SED-Lab.git`

3) cd (go) to the project directory run code:
<br>
`cd SED-Lab`

4) Create Virtual environment:
<br>
Open terminal in the project folder.
<br>
`python -m venv venv`

5) Activate virtual env.
<br>
`venv\Scripts\activate`

6) Update pip:
<br>
`python -m pip install --upgrade pip`

7) Install dependencies:
<br>
`pip install -r requirements.txt`

## How to use?
This is all-in-one SED (Sound Event Detection) tasks repo. It's meant to be adapted and optimized for specific SED task. Experiment to find the best configuration for your task.

### Prepare
Prepare your local workspace, run: `python src/utils/prep.py`
<br>Place your dataset data in `DATA/DATASET` directory. Audio samples are in <b>.wav</b> format and <b>every subdirectory in DATASET directory is an unique label</b>.
<br>If needed, use [Audio-Dataset-Recorder-Tools](https://github.com/Teapack1/Audio-Dataset-Recorder-Tools) for new recordings.

### Training

Open notebook located in the notebooks directory `training_notebook.ipynb`.
<br>Setup the first cell "Training Parameters" according to your task.
<br>In the training notebook press "run all", wait until it finishes.
<br>You get `MODEL`, `ENCODER`, `SCALER`, `CONFIG` and various plots from training and evaluation.
- MODEL: Trained tf/tflite model of [custom or preset architecture](src/model/model.md).
- ENCODER: One-hot label encoder saved as joblib.
- SCALER: Selected scaler fitted on the training dataset saved as joblib.
- CONFIG: Parameters from the training are saved as json to be used in the inference script.
- PLOTS: In plots directory `MODEL/PLOTS` are saved training plots and evaluation conusion matrix.

### Inference

From `MODEL` directory, copy <b>CONFIG</b> file of desired trained model, put it in `config` folder, and rename to `config.json`.
<br>Edit or just run this inference code in terminal: `python inference.py -conf MODEL/config.json -prob 0.9 -o 0.2 -ac 0 -au 0 -aip "127.0.0.1"`

If you did not train model with this repo, you need to setup inference via arguents.


## Arguments and Settings
This contains documentation of possible settings in training and inference scripts.

### Training Settings
Open `training_notebook.ipynb` in notebooks directory and setup training parameters at the top notebook cell.
#### Audio Samples:
- **`AUDIO_CHUNK:`** Sets uniform length of the audio samples for training. For the training, all samples need to be the same size. The sample length is set to 0.4 seconds.
- **`SLICE_AUDIO:`** Determines whether to trim/pad audio samples to the `AUDIO_CHUNK` length. Not necessary when dataset is recorded in uniform audio length. 
- **`DATA_RANGE:`** Specifies the range of data values that the neural network will take. It can be 1 or 255. Ranges tensor values 0-1 or 0-255. Some models require 255.
- **`NUM_CHANNELS:`** The number of audio channels. Typically 1 for mono audio.
- **`SAMPLE_RATE:`** The sample rate in Hz. 44100 Hz is ok for sounds.

#### Audio Feature Parameters:
- **`MAIN_FEATURE:`** Specifies the main feature type for audio processing. Can be 'mfcc', 'mel', or 'stft'. Meaning Mel Frequency Cepstral Coefficients (MFCC), Mel Spectrograms, or Short-Time Fourier Transform (STFT) spectrograms.
- **`N_MELS:`** Number of Mel bands, applicable only when `MAIN_FEATURE` is set to 'mel'.
- **`NFFT:`** Size of a window (frame) of the audio in number of samples, on which the <i>fourier transform</i> is proceeded.
- **`HOP_LENGTH:`** Number of samples between successive frames.
- **`N_MFCC:`** Number of MFCC coefficients, applicable only when `MAIN_FEATURE` is set to 'mfcc'.
- **`FMAX:`** Maximum frequency for the <i>FFT</i> to capture, typically the Nyquist frequency to avoid aliasing (half of `SAMPLE_RATE`).
- **`N_FRAMES:`** Number of frames for the audio feature to have. It dynamically adjusts based on your <i>sample rate, audio chunk, hop length.</i>
- **`SCALER:`** Data normalization method. Will be fitted on the whole dataset and then saved. Can be 'standard', 'minmax', 'robust', 'maxabs', or 'None'. Minmax worked good for spectrograms but try experimenting here.
#### Training Settings:
- **`EPOCHS:`** Number of epochs for training. With datasets around 10k items, features apx. 100x100x1 and lr 1e-3 i go for 50 epochs.
- **`BATCH_SIZE:`** The size of batches used in training. How many audio files does the network take at once, before it calculates gradient descent and updates its weights. Usually go with 16. Bigger might not learn to generalize that well, too small might become unstable gradients.
- **`LEARNING:RATE:`** Speed at which the network learns, start with 1e-3, when it the training fluctuates too much , reduce to 1e-4 and increase epochs.
- **`AUGMENTATION`** Do you want to use augmentations in the training ? Width and height shift in modest rate is applied. 

#### Model Settings:
- **`MODEL_FORMAT:`** Format of the output model. Can be 'h5', 'keras', 'tf', or 'tflite'. Keras is a modern format, well optimized.
- **`LITE_VERSION:`** Indicates whether to produce a TensorFlow Lite version of the model.
- **`MODEL_ARCH:`** The architecture of the model to be used. Currently set to "SmallerVGGNet".
- **`MODEL_TYPE:`** The type of model. Currently set to "default".
- **`NEW_MODEL_NAME:`** Setup your custom name for this session outputs. It will <b>keep you organised</b> and marks every output from this session with this label.

### Inference Arguments

Inference arguments are passed when calling the `inference.py` script. 
<br>Example: `python inference.py -conf MODEL/config.json -prob 0.9 -tw "music, children, saw" -o 0.1 -ac 0 -au 0 -aip "127.0.0.1"`

For running the inference script, several parameters are required. Most of these parameters are stored in the `config.json` file, which is generated during the training phase. This file includes settings about the audio processing, features and model, it helps automate the setup process, reducing the need for manual input. Below is a description of the inference arguments:

#### Configuration File:
- **`-conf, --config_path:`** Path to the configuration file (`config.json`). This file includes most of the necessary parameters for inference, such as model paths, feature extraction settings, and more.
  
#### Prediction and Thresholds:
- **`-prob, --probability_threshold:`** Confidence threshold for predictions. (default: 0.9)
- **`-o, --chunk_overlap:`** Realtime recording sample overlap in seconds. (default: 0.2)
- **`-tw, --trigger_words:`** List of labels that will trigger the action. `config.json` can be overriden by argument to select just some labels.

#### Art-Net Configuration:
- **`-ac, --artnet_channel:`** Art-Net channel to send data to. (default: 0)
- **`-au, --artnet_universe:`** Art-Net universe to send data to. (default: 0)
- **`-aip, --artnet_ip:`** IP address of the Art-Net node. (default: "127.0.0.1")

#### Device Configuration:
- **`-dev, --device:`** Processing unit to use. Options are "cpu" or "gpu". (default: "cpu")
- **`-mic, --mic_device:`** Microphone device index. (default: 0)

### Parameters for Manual Setup (if config.json is not used):
If the `config.json` file is not available, the following parameters need to be set manually:

- **`--model_path:`** Path to the trained model file.
- **`--labeler_path:`** Path to the labeler file.
- **`--lite_model_path:`** Path to the trained Lite model file. 
- **`--scaler_path:`** Path to the feature scaler file.
- **`--sample_rate:`** Audio sample rate (default: 44100).
- **`--num_channels:`** Number of audio channels (default: 1).
- **`--audio_chunk:`** Length of audio slice in seconds (default: 0.4).
- **`--num_mels:`** Number of Mel bands to generate (default: 256).
- **`--n_fft:`** Number of samples in each FFT window (default: 2048).
- **`--fmax:`** Maximum frequency when computing MEL spectrograms (default: 22050).
- **`--hop_length:`** Number of samples between successive FFT windows (default: 512).
- **`--n_frames:`** Number of frames of audio to use for prediction (default: 34).
- **`--data_range:`** Range of data values (1 or 255, default: 255).
- **`--n_mfcc:`** Number of MFCCs to extract (default: 40).

## Recommended Application Settings

- **Speech Commands:** `SAMPLE_RATE`: 16000, `N_MELS`: 40, `MAIN_FEATURE`: mfcc, `NFFT`: 512, `HOP_LENGTH`: 256, `N_MFCC`: 13, `AUDIO_CHUNK`: 1.0, `SCALER`: minmax
  
- **Music Genre Classification:** `SAMPLE_RATE`: 22050, `N_MELS`: 128, `MAIN_FEATURE`: mel, `NFFT`: 2048, `HOP_LENGTH`: 512, `AUDIO_CHUNK`: 3.0, `SCALER`: standard
  
- **Cheering Glasses:** `SAMPLE_RATE`: 44100, `N_MELS`: 64, `MAIN_FEATURE`: mel, `NFFT`: 1024, `HOP_LENGTH`: 512, `AUDIO_CHUNK`: 0.6, `SCALER`: robust
  
- **Bird Song Recognition:** `SAMPLE_RATE`: 48000, `N_MELS`: 128, `MAIN_FEATURE`: mel, `NFFT`: 1024, `HOP_LENGTH`: 256, `AUDIO_CHUNK`: 5.0, `SCALER`: minmax
  
- **Heartbeat Sound Detection:** `SAMPLE_RATE`: 8000, `N_MELS`: 40, `MAIN_FEATURE`: stft, `NFFT`: 256, `HOP_LENGTH`: 128, `AUDIO_CHUNK`: 1.0, `SCALER`: standard
  
- **Urban Sound Classification:** `SAMPLE_RATE`: 22050, `N_MELS`: 64, `MAIN_FEATURE`: mel, `NFFT`: 1024, `HOP_LENGTH`: 512, `AUDIO_CHUNK`: 4.0, `SCALER`: minmax

## Results
Neural networks consisting of three CNN blocks trained with default parameters on custom datasets of approximately 10,000 samples and up to 5 categories typically achieve high accuracy, often surpassing 95% in classification tasks. 
<br>While niche tasks can benefit from fine-tuning training parameters and feature engineering, increasing the dataset size and variety, as well as maintaining balanced categories, consistently yields the best results for model generalization.

![Sound Event Detection and Audio Classification](/src/img/accPlot.png)

## License
This project is licensed under the Apache 2 License - see the [LICENSE.md](LICENSE) file for details.

---
