# %% [markdown]
# # 1. Import and Install Dependencies

# %% [markdown]
# ## 1.1 Install Dependencies

# %%
#!pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 tensorflow-io matplotlib==3.7.*

# %% [markdown]
# ## 1.2 Load Dependencies

# %%
import os
from matplotlib import pyplot as plt
import tensorflow as tf 
from classify_utilities import AudioProcessor
import pandas as pd
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler
import librosa
import numpy as np
import joblib
import random

# %%
print(tf.config.list_physical_devices('GPU'))

# %% [markdown]
# # 2. Preprocess the Data

# %% [markdown]
# ## 2.1 Define Paths to Files

# %%
DATA_DIR = 'data'
METADATA = os.path.join("metadata.csv")
MODEL_PATH = os.path.join("model", "model.keras")
LABELER_PATH = os.path.join("model", "label_encoder.joblib")
CAPUCHIN_FILE = os.path.join(DATA_DIR, 'Parsed_Capuchinbird_Clips', 'XC3776-3.wav')
NOT_CAPUCHIN_FILE = os.path.join(DATA_DIR, 'Parsed_Not_Capuchinbird_Clips', 'afternoon-birds-song-in-forest-0.wav'),

# %%
SLICE_LENGTH = 3 # seconds
NUM_CHANNELS = 1
SAMPLE_RATE = 22050

N_MELS = 128
NFFT = 2048
FMAX = 11000
HOP_LENGTH = 512

EPOCHS = 10
BATCH_SIZE = 32

# %%
audio_processor = AudioProcessor(sample_rate=SAMPLE_RATE, 
                                 n_mels = N_MELS,
                                 fmax = FMAX,
                                 n_fft = NFFT,
                                 hop_length = HOP_LENGTH, 
                                 slice_length = SLICE_LENGTH,
                                 )
                

# %% [markdown]
# ## 2.2 Label Encoding

# %%
label_encoder = OneHotEncoder()

classes = os.listdir(DATA_DIR)
classes.sort()
classes = np.array(classes).reshape(-1, 1)

label_encoder.fit(classes)
#labels = label_encoder.transform(classes).toarray()
#original_data = label_encoder.inverse_transform(labels)

# Serialize and save the fitted encoder
joblib.dump(label_encoder, LABELER_PATH)

def idx2label(idx):
    idx_reshaped = np.array(idx).reshape(1, -1)
    return label_encoder.inverse_transform(idx_reshaped)[0][0]

def label2idx(label):
    label = np.array(label).reshape(-1, 1)
    return label_encoder.transform(label).toarray()[0]

# %%
label2idx('Parsed_Capuchinbird_Clips')

# %%
idx2label(label2idx('Parsed_Capuchinbird_Clips'))

# %%
# External labeler
audio_processor.idx2label([1., 0., 0.], joblib.load(LABELER_PATH))

# %% [markdown]
# ## 2.3 Exploratory data analysis

# %% [markdown]
# ### Produce metadata dataframe

# %%
# Analyze dataset:
# List all the files in dictionare and subdictionaries.
metadata = []

for root, _, files in os.walk(DATA_DIR):
    for i, file in enumerate(files):
        if file.endswith('.wav'):
            filename = os.path.join(root, file)
            label = os.path.basename(root)
            class_ = label2idx(label)
            num_channels, sample_rate, bit_depth, avg_rms, length_in_seconds, length_in_frames = audio_processor.read_file_properties(filename)
            metadata.append({
                'filename': filename, 
                'label': label, 
                'class': class_,
                'num_channels': num_channels, 
                'sample_rate': sample_rate, 
                'bit_depth': bit_depth, 
                'avg_rms': avg_rms, 
                'length_in_seconds': length_in_seconds, 
                'length_in_frames': length_in_frames
            })

            print(f"Processed {i} file. {file}")
        else:
            print(f"Skipped {i} file. {file}")
            
metadata = pd.DataFrame(metadata)
metadata.to_csv(METADATA, index=False)

# %% [markdown]
# ### Observe the data

# %%
metadata.head()

# %% [markdown]
# ### class balance

# %%
print(metadata["label"].value_counts())

# %% [markdown]
# ### plot class waveforms

# %%
labels = metadata["label"].unique()

fig = plt.figure(figsize=(8,8))

fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i, label in enumerate(labels):
    filtered_df = metadata[metadata["label"] == label]
    slice_file_name = filtered_df["filename"].iloc[0]
    fold = filtered_df["label"].iloc[0]
    fig.add_subplot(5, 2, i+1)
    plt.title(label)
    data, sr = librosa.load(os.path.join(slice_file_name))
    librosa.display.waveshow(y = data, sr=sr, color="r", alpha=0.5, label='Harmonic')
    print(slice_file_name)
     
plt.tight_layout()  # This will adjust spacing between subplots to prevent overlap
plt.show()  # This will display the plot

# %%
# num of channels 
print("Channels: ")
print(metadata.num_channels.value_counts(normalize=True))
print("\n")

# sample rates 
print("Sample Rates: ")
print(metadata.sample_rate.value_counts(normalize=True))
print("\n")

# bit depth
print("Bit Depth: ")
print(metadata.bit_depth.value_counts(normalize=True))
print("\n")

# length in samples
print("Samples: ")
print(metadata.length_in_frames.describe())
print("\n")

# length in seconds
print("Length (s): ")
print(metadata.length_in_seconds.describe())

# RMS
# print(audiodf.avg_rms.describe())


# %% [markdown]
# ## 2.4 Audio Preprocessing:

# %% [markdown]
# ### Test audio preprocessing methods
# -   Librosa -> 
#         SR: 22050
#         channel: 1
#     trim/pad ->
#         length: 3s (3x22050)
#     spectrogram ->
#         mel-spectrogram / spectrogram / MFCC
#     post-process ->
#         to Db (log scale, more apparent patterns)
#         abs 
#     normalize ->
#         
#         
# 

# %%
# Test the Fourier transform
#In each iteration of the loop, the variable index is assigned the index value of the current row, and the variable row is assigned the data of the current row (as a Series object).
rows = metadata.iloc[[0, 1]]
#rows = metadata.sample(2)
slice_length = 3
samples_show = len(rows)
pass_ = 0

fig, axes = plt.subplots(nrows=samples_show, ncols=2, figsize=(12, samples_show* 3.6))

for i, row in rows.iterrows():    
    if pass_ > samples_show:
        break
    audio_file, librosa_sample_rate = librosa.load(row["filename"], sr=SAMPLE_RATE)
    if slice_length: 
        sample_length = slice_length * librosa_sample_rate

        audio_file = audio_file[:sample_length]
        if len(audio_file) < sample_length:
            audio_file = np.pad(audio_file, (0, sample_length - len(audio_file)), constant_values=0)

    spectrogram = librosa.feature.melspectrogram(y=audio_file, sr=librosa_sample_rate, n_mels=128, fmax = 11000, n_fft=2048, hop_length=512)
    spectrogram = (librosa.power_to_db(spectrogram, ref=np.max))
    spectrogram = np.abs(spectrogram)

    #scaler = StandardScaler()
    #spectrogram = scaler.fit_transform(spectrogram.T)
    #spectrogram = spectrogram.T # transpose back to original shape for display purpocess

        # wave Plot
    axes[pass_, 0].set_title(f"Label: {row['label']} Waveform")
    librosa.display.waveshow(audio_file, sr=librosa_sample_rate, ax=axes[pass_, 0])
    # spectrogram plot
    axes[pass_, 1].set_title(f"Label: {row['label']} Spectrogram")
    img = librosa.display.specshow(spectrogram, sr=librosa_sample_rate, x_axis='time', y_axis='mel', ax=axes[pass_, 1])
    pass_ += 1


print(f"audio_file shape {audio_file.shape} - (frames, channels)")
print(f"audio_file sample rate {librosa_sample_rate} Hz")
print(f"Spectrogram shape {spectrogram.shape} - (mels/frequency, frames/time)")
print(f"spectrogram min: {spectrogram.min()} spectrogram max: {spectrogram.max()}, average: {spectrogram.mean()}")
print(f"spectrogram dtype: {spectrogram.dtype}")
print(f"audio dtype: {audio_file.dtype} - bit depth")

#fig.colorbar(img, ax=axes[:, 0], format='%+2.0f dB')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Preprocessing function

# %%
def extract_features(row):
    
    class_label = row["class"]
    
    audio_file,_ = librosa.load(row["filename"], sr=SAMPLE_RATE)

    spectrogram = audio_processor(
        data = audio_file)
    
    shape = spectrogram.shape

    return spectrogram, class_label, shape

# %% [markdown]
# ## 3 Produce Dataset

# %% [markdown]
# ## 3.1 Extract features and labels into dataframe

# %%
#In each iteration of the loop, the variable index is assigned the index value of the current row, and the variable row is assigned the data of the current row (as a Series object).
features = []

for index, row in metadata.iterrows():
    features.append(extract_features(row))
    print(f"Processed {index} file. {row['filename']}")
    
dataset_df = pd.DataFrame(features, columns=["features", "class_label", "shape"])
print('Finished feature extraction from ', len(dataset_df), ' files') 

# %%
dataset_df.head()

# %%
dataset_df["shape"].value_counts()

# %% [markdown]
# ## 3.2 Train Test Split

# %%
X = np.array(dataset_df.features.tolist())
y = np.array(dataset_df.class_label.tolist())
X.shape

# %%
print(f"randomm feature example: {X[0]} and label: {y[0]}")

# %%
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state = 42)

# %% [markdown]
# ## 3.3 Reshape sets for NN input layer

# %%
x_train = x_train.reshape(x_train.shape[0], 130, N_MELS, NUM_CHANNELS)
x_test = x_test.reshape(x_test.shape[0], 130, N_MELS, NUM_CHANNELS)

num_labels = y.shape[1]
print(f"num_labels: {num_labels}")
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# %% [markdown]
# # 4. Build Deep Learning Model

# %% [markdown]
# ## 4.1 Load Tensorflow Dependencies

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall

# %% [markdown]
# ## 7.2 Build Sequential Model

# %%
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(130, N_MELS, NUM_CHANNELS)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_labels, activation='softmax'))

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# %%
model.compile(
    optimizer=optimizer, 
    loss='CategoricalCrossentropy', 
    metrics=['accuracy', Precision(), Recall()]
    )

# %%
model.summary()

# %% [markdown]
# ## 4.3 Fit Model, View Loss and KPI Plots

# %%
hist = model.fit(x_train, 
                 y_train, 
                 epochs=EPOCHS, 
                 validation_data=(x_test, y_test), 
                 batch_size=BATCH_SIZE
                 )

# %%
plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.show()

# %%
plt.title('Accuracy')
plt.plot(hist.history['accuracy'], 'r')
plt.plot(hist.history['val_accuracy'], 'b')
plt.show()

# %%
plt.title('Precision')
plt.plot(hist.history['precision'], 'r')
plt.plot(hist.history['val_precision'], 'b')
plt.show()

# %%
plt.title('Recall')
plt.plot(hist.history['recall'], 'r')
plt.plot(hist.history['val_recall'], 'b')
plt.show()

# %% [markdown]
# # 5. Make a Prediction on a Single Clip

# %% [markdown]
# ## 5.1 Make a Prediction, Evaluate

# %%
predictions = model.predict(x_test)

# %%
print(idx2label(predictions[2]))
print(idx2label(y_test[21]))
print(f"input default shape: {x_test[1].shape}")
print(f"reshaped input feature shape: {np.expand_dims((x_test[21]), axis=0).shape}")

# %%
prediction = model.predict(np.expand_dims((x_test[21]), axis=0))
print(prediction)
print(idx2label(prediction))

# %%
# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=1)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=1)
print("Testing Accuracy: ", score[1])

# %% [markdown]
# ## 5.2 Save Model

# %%
model.save(MODEL_PATH)

# %% [markdown]
# # 6. Inference 

# %% [markdown]
# ## 6.1 Load local model and labels

# %%
# Load the encoder in the inference environment
loaded_encoder = joblib.load(LABELER_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# %% [markdown]
# ## 6.1 Inference on loacl files

# %%
test_data_dir = os.path.join('test')

audio_files = os.listdir(test_data_dir)
random.shuffle(audio_files)

print(audio_files[1])
audio_labels = [os.path.splitext(file)[0] for file in audio_files]

# %%
for file in audio_files:
    path = os.path.join(test_data_dir, file)
    print(path)
    data, _ = librosa.load(path, sr=SAMPLE_RATE)
    prediction_feature = audio_processor(
            data = data
        )
    
    # Reshape to match model input shape
    prediction_feature = prediction_feature.reshape(1, 130, N_MELS, NUM_CHANNELS)
    predicted_class = idx2label(model.predict(prediction_feature)) 
    print("The predicted class is:", predicted_class, '\n') 



# %% [markdown]
# ## 6.2 Real-time inference

# %% [markdown]
# ## run "run.py"

# %%
from inference import SoundClassificationService

def main():

    config = {
        "model_path": MODEL_PATH,
        "labels_path": LABELER_PATH,
        
        "sample_rate": SAMPLE_RATE,
        "num_channels": NUM_CHANNELS,
        "slice_length": SLICE_LENGTH,
        
        "num_mels": N_MELS,
        "n_fft": NFFT,
        "fmax": FMAX,
        "hop_length": HOP_LENGTH,
        
        "confidence_threshold": 0.5,
        "listening_hop_length": 0.6,
        "device": "cpu"

    }

    service = SoundClassificationService.get_instance(config)
    service.listen_and_predict(duration=SLICE_LENGTH, overlap=0.5)


if __name__ == "__main__":
    main()

# %%



