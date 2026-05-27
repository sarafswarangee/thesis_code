import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from iqtools import TIQData


FILE_LIST = [
    "data/IQDataFile-2024.04.18.19.20.44.849.tiq",
    "data/IQDataFile-2024.04.18.19.20.28.613.tiq",
    "data/IQDataFile-2024.04.18.19.22.40.034.tiq",
    "data/IQDataFile-2024.04.18.19.20.36.730.tiq",
    "data/IQDataFile-2024.04.18.19.20.04.270.tiq",
    "data/IQDataFile-2024.04.18.19.20.12.385.tiq",
    "data/IQDataFile-2024.04.18.19.20.20.500.tiq",
]

CHUNK_SIZE = 1024
NAVG = 1
LOW_FREQ = 0.8e6
HIGH_FREQ = 1.86e6
RANDOM_STATE = 21
TEST_SIZE = 0.2
GROUP_SIZE = 1000
WEIGHTS_PATH = "models/1024FWHM8files.weights.h5"


def get_sampling_rate(filename):
    data = TIQData(filename)
    data.read_samples(112000000, 0)
    return data.fs


def create_band_mask(chunk_size, fs, low_freq, high_freq):
    freqs = np.fft.fftshift(np.fft.fftfreq(chunk_size, d=1 / fs))
    return (freqs >= low_freq) & (freqs <= high_freq)


def load_psd_dataset(filenames, chunk_size, fs, band_mask, navg=1):
    window = np.hanning(chunk_size)
    psd_samples = []
    block_size = chunk_size * navg

    for filename in filenames:
        data = TIQData(filename)
        data.read_samples(112000000, 0)

        iq = data.data_array
        if iq.ndim == 2:
            iq = iq[0]

        usable_samples = (len(iq) // block_size) * block_size
        iq = iq[:usable_samples]

        blocks = iq.reshape(-1, block_size)

        for block in blocks:
            subblocks = block.reshape(navg, chunk_size)
            windowed = subblocks * window

            fft_blocks = np.fft.fftshift(
                np.fft.fft(windowed, axis=1),
                axes=1,
            )

            selected_fft = fft_blocks[:, band_mask]
            psd = np.mean(np.abs(selected_fft) ** 2, axis=0)

            psd_samples.append(psd)

    return np.stack(psd_samples)


def prepare_data(data, test_size=0.2, random_state=21):
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )

    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    return (
        tf.cast(train_data, tf.float32),
        tf.cast(test_data, tf.float32),
    )


def create_group_mean_targets(train_data, group_size=1000):
    train_data_np = train_data.numpy()
    targets = np.zeros_like(train_data_np)

    for i in range(train_data_np.shape[0]):
        group_start = (i // group_size) * group_size
        group_end = min(group_start + group_size, train_data_np.shape[0])
        targets[i] = train_data_np[group_start:group_end].mean(axis=0)

    return targets


class CNNAnomalyDetector(Model):
    def __init__(self, input_shape=(78, 1)):
        super().__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(1, 3, activation="relu", padding="valid"),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(8, 3, activation="relu", padding="same"),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(16, 3, activation="relu", padding="same"),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(8, 3, activation="relu", padding="same"),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(4, 3, activation="relu", padding="same"),
            layers.MaxPooling1D(2, padding="same"),
            layers.Flatten(),
            layers.Dense(1, activation="relu"),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(1,)),
            layers.Dense(16, activation="relu"),
            layers.Reshape((4, 4)),
            layers.Conv1D(4, 3, activation="relu", padding="same"),
            layers.UpSampling1D(2),
            layers.Conv1D(8, 3, activation="relu", padding="same"),
            layers.UpSampling1D(2),
            layers.Conv1D(16, 3, activation="relu", padding="same"),
            layers.UpSampling1D(2),
            layers.Conv1D(8, 3, activation="relu", padding="same"),
            layers.UpSampling1D(2),
            layers.Conv1D(8, 3, activation="relu", padding="same"),
            layers.UpSampling1D(2),
            layers.Conv1D(1, 3, activation="sigmoid", padding="same"),
            layers.Lambda(lambda x: x[:, :78, :]),
        ])

    def call(self, x):
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return tf.squeeze(decoded, axis=-1)


def load_model(weights_path):
    model = CNNAnomalyDetector()
    model(tf.zeros((1, 78)))
    model.load_weights(weights_path)
    model.compile(optimizer="adam", loss="mse")
    return model


if __name__ == "__main__":
    fs = get_sampling_rate(FILE_LIST[0])
    band_mask = create_band_mask(CHUNK_SIZE, fs, LOW_FREQ, HIGH_FREQ)

    data = load_psd_dataset(
        FILE_LIST,
        CHUNK_SIZE,
        fs,
        band_mask,
        navg=NAVG,
    )

    train_data, test_data = prepare_data(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_targets = create_group_mean_targets(
        train_data,
        group_size=GROUP_SIZE,
    )

    autoencoder = load_model(WEIGHTS_PATH)

    print("Dataset shape:", data.shape)
    print("Training data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    print("Training targets shape:", train_targets.shape)