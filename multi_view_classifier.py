import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from iqtools import TIQData


SEED = 123
CHUNK_SIZE = 1024
NAVG = 1
FS_SIGNAL = 14_000_000
LOW_FREQ = 0.8e6
HIGH_FREQ = 1.86e6
SAMPLE_COUNT = 112_000_000

AE_WEIGHTS_PATH = "models/1024FWHM4_cav.weights.h5"

NOISE_FILES = [
    "data/noise/IQDataFile-2024.04.18.19.20.44.849.tiq",
    "data/noise/IQDataFile-2024.04.18.19.20.28.613.tiq",
    "data/noise/IQDataFile-2024.04.18.19.22.40.034.tiq",
    "data/noise/IQDataFile-2024.04.18.19.20.36.730.tiq",
]

SIGNAL_FILES = [
    "data/signals/IQData_with_8.1GHz_Chirpe13.bin",
    "data/signals/IQData_with_8.1GHz_Chirpe12.bin",
    "data/signals/IQData_with_8.1GHz_Chirpe10.bin",
    "data/signals/IQData_with_8.1GHz_Chirpe11.bin",
]


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_band_mask(chunk_size, fs, low_freq, high_freq):
    freqs = np.fft.fftshift(np.fft.fftfreq(chunk_size, d=1 / fs))
    return (freqs >= low_freq) & (freqs <= high_freq)


def get_tiq_sampling_rate(filename):
    data = TIQData(filename)
    data.read_samples(SAMPLE_COUNT, 0)
    return data.fs


def read_tiq_file(filename):
    data = TIQData(filename)
    data.read_samples(SAMPLE_COUNT, 0)

    iq = data.data_array
    if iq.ndim == 2:
        iq = iq[0]

    return iq


def read_binary_iq_file(filename):
    return np.fromfile(filename, dtype=np.complex128, count=SAMPLE_COUNT)


def load_psd_dataset(filenames, chunk_size, fs, band_mask, reader, navg=1):
    window = np.hanning(chunk_size)
    block_size = chunk_size * navg
    psd_samples = []

    for filename in filenames:
        iq = reader(filename)

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


def normalize_with_reference(data, min_value, max_value):
    normalized = (data - min_value) / (max_value - min_value)
    return tf.cast(normalized, tf.float32)


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


def load_autoencoder(weights_path):
    autoencoder = CNNAnomalyDetector()
    autoencoder(tf.zeros((1, 78)))
    autoencoder.load_weights(weights_path)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


class MultiViewPSDClassifier(Model):
    def __init__(self, psd_shape=(78,), recon_shape=(78,), normalize=True):
        super().__init__()

        self.normalize = normalize

        if normalize:
            self.psd_norm = layers.Normalization(axis=-1)
            self.recon_norm = layers.Normalization(axis=-1)
        else:
            self.psd_norm = layers.Lambda(lambda x: x)
            self.recon_norm = layers.Lambda(lambda x: x)

        self.psd_branch = tf.keras.Sequential([
            layers.Input(shape=psd_shape),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
        ])

        self.recon_branch = tf.keras.Sequential([
            layers.Input(shape=recon_shape),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
        ])

        self.error_branch = tf.keras.Sequential([
            layers.Input(shape=(1,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
        ])

        self.classifier = tf.keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(1, activation="sigmoid"),
        ])

    def adapt(self, psd_data, recon_data):
        if self.normalize:
            self.psd_norm.adapt(psd_data)
            self.recon_norm.adapt(recon_data)

    def call(self, inputs):
        psd, reconstruction, error_score = inputs

        if len(error_score.shape) == 1:
            error_score = tf.expand_dims(error_score, axis=-1)

        psd_embedding = self.psd_branch(self.psd_norm(psd))
        recon_embedding = self.recon_branch(self.recon_norm(reconstruction))
        error_embedding = self.error_branch(error_score)

        features = tf.concat(
            [psd_embedding, recon_embedding, error_embedding],
            axis=-1,
        )

        return self.classifier(features)


def build_classifier_dataset(noise_psd, signal_psd, autoencoder):
    min_value = tf.reduce_min(noise_psd)
    max_value = tf.reduce_max(noise_psd)

    noise_data = normalize_with_reference(noise_psd, min_value, max_value)
    signal_data = normalize_with_reference(signal_psd, min_value, max_value)

    noise_recon = autoencoder(noise_data).numpy()
    signal_recon = autoencoder(signal_data).numpy()

    noise_data = noise_data.numpy()
    signal_data = signal_data.numpy()

    x_psd = np.concatenate([signal_data, noise_data], axis=0)
    x_recon = np.concatenate([signal_recon, noise_recon], axis=0)

    y_signal = np.ones(len(signal_data), dtype=int)
    y_noise = np.zeros(len(noise_data), dtype=int)
    y = np.concatenate([y_signal, y_noise], axis=0)

    error_score = np.sum(np.abs(x_psd - x_recon), axis=1).reshape(-1, 1)

    indices = np.random.permutation(len(x_psd))

    return (
        x_psd[indices],
        x_recon[indices],
        error_score[indices],
        y[indices],
    )


if __name__ == "__main__":
    set_seed(SEED)

    noise_fs = get_tiq_sampling_rate(NOISE_FILES[0])
    noise_band_mask = create_band_mask(
        CHUNK_SIZE,
        noise_fs,
        LOW_FREQ,
        HIGH_FREQ,
    )

    signal_band_mask = create_band_mask(
        CHUNK_SIZE,
        FS_SIGNAL,
        LOW_FREQ,
        HIGH_FREQ,
    )

    noise_psd = load_psd_dataset(
        NOISE_FILES,
        CHUNK_SIZE,
        noise_fs,
        noise_band_mask,
        reader=read_tiq_file,
        navg=NAVG,
    )

    signal_psd = load_psd_dataset(
        SIGNAL_FILES,
        CHUNK_SIZE,
        FS_SIGNAL,
        signal_band_mask,
        reader=read_binary_iq_file,
        navg=NAVG,
    )

    autoencoder = load_autoencoder(AE_WEIGHTS_PATH)

    x_psd, x_recon, x_error, y = build_classifier_dataset(
        noise_psd,
        signal_psd,
        autoencoder,
    )

    (
        x_psd_train,
        x_psd_val,
        x_recon_train,
        x_recon_val,
        x_error_train,
        x_error_val,
        y_train,
        y_val,
    ) = train_test_split(
        x_psd,
        x_recon,
        x_error,
        y,
        test_size=0.1,
        stratify=y,
        random_state=42,
    )

    classifier = MultiViewPSDClassifier(
        psd_shape=(78,),
        recon_shape=(78,),
        normalize=True,
    )

    classifier.adapt(x_psd_train, x_recon_train)

    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    classifier.fit(
        [x_psd_train, x_recon_train, x_error_train],
        y_train,
        validation_data=(
            [x_psd_val, x_recon_val, x_error_val],
            y_val,
        ),
        epochs=30,
        batch_size=32,
    )