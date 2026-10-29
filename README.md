
# High-Frequency Gravitational-Wave Detection
### CNN Autoencoder and Multi-View Signal Classification

Code accompanying **Swarangee Saraf's Master's thesis** at the **University of Bonn**.

This repository contains Python implementations of two machine-learning components developed for the analysis of frequency-domain measurements in a high-frequency gravitational-wave search: a convolutional autoencoder for reconstructing noise spectra and a multi-view classifier that distinguishes noise-only from signal-containing examples.

**Thesis:** [Read the Master's thesis (PDF)](Saraf_Swarangee_Master_s_Thesis.pdf)

## What's in this repository?

| File | Purpose |
| --- | --- |
| [`CNN_autoencoder.py`](CNN_autoencoder.py) | Defines the convolutional autoencoder, processes `.tiq` I/Q recordings into spectral-power vectors, and loads saved autoencoder weights. |
| [`multi_view_classifier.py`](multi_view_classifier.py) | Builds noise/signal datasets, obtains autoencoder reconstructions and reconstruction-error features, and trains a binary classifier. |
| [`Model Weights/`](Model%20Weights/) | Contains two saved autoencoder weight files (`.weights.h5`). |
| [`Saraf_Swarangee_Master_s_Thesis.pdf`](Saraf_Swarangee_Master_s_Thesis.pdf) | The accompanying thesis. |

## Method overview

Both scripts process complex I/Q samples in **1,024-sample windows**. They apply a Hann window, compute a centered FFT, and select frequencies between **0.8 MHz and 1.86 MHz** in the recorded signal's frequency representation. The resulting features are squared FFT magnitudes, averaged across subwindows when applicable (`NAVG = 1` in the checked-in configuration). The neural-network implementations expect **78 features per example**.

**CNN autoencoder.** The `CNNAnomalyDetector` model has convolutional encoder and decoder stages. The script prepares training and test spectra, normalizes both using the training-data minimum and maximum, constructs group-mean targets, and **loads previously saved weights**. Its current main block prints dataset and split shapes; it does not train the autoencoder or produce detection metrics.

**Multi-view classifier.** The `MultiViewPSDClassifier` combines three inputs: the spectral-power vector, its autoencoder reconstruction, and a scalar reconstruction-error score (the sum of absolute differences). It uses separate neural-network branches for these inputs, concatenates their learned features, and predicts a binary label: `0` for noise-only data and `1` for signal-containing data. The script uses a stratified 90/10 train/validation split and trains for **30 epochs**, with **batch size 32**.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/sarafswarangee/thesis_code.git
cd thesis_code
```

### 2. Install Python dependencies

The scripts import **NumPy**, **TensorFlow/Keras**, **scikit-learn**, and **iqtools** (`TIQData`). You can create a Python virtual environment and install these packages:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
python -m pip install numpy tensorflow scikit-learn iqtools
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. No pinned dependency versions or tested environment specification are currently included, so package compatibility may depend on your Python and TensorFlow versions.

### 3. Supply the input recordings

The raw input files are **not included** in this repository. Both scripts currently refer to fixed file paths:

- `CNN_autoencoder.py` expects seven `.tiq` recordings directly inside `data/`, with exact filenames specified in `FILE_LIST`.
- `multi_view_classifier.py` expects four `.tiq` recordings in `data/noise/` (`NOISE_FILES`) and four signal-containing complex I/Q files in `data/signals/` (`SIGNAL_FILES`).
- The classifier reads its signal-containing `.bin` files as `complex128` values. The current names are `IQData_with_8.1GHz_Chirpe13.bin`, `IQData_with_8.1GHz_Chirpe12.bin`, `IQData_with_8.1GHz_Chirpe10.bin`, and `IQData_with_8.1GHz_Chirpe11.bin`.

Place the appropriate files at those locations, or edit the path lists at the top of the scripts to match your local dataset. The scripts request **112,000,000 I/Q samples per input file**; make sure adequate memory is available.

### 4. Set the model-weight paths

The repository stores the pretrained weights under `Model Weights/`, but the scripts look under `models/`. **Before running**, either update the `WEIGHTS_PATH` and `AE_WEIGHTS_PATH` variables, or create a `models/` folder and copy these files into it:

| Script | Required weight file |
| --- | --- |
| `CNN_autoencoder.py` | `models/1024FWHM8files.weights.h5` |
| `multi_view_classifier.py` | `models/1024FWHM4_cav.weights.h5` |

The original files are in [`Model Weights/`](Model%20Weights/).

### 5. Run the scripts

From the repository root, after providing the data and correcting the weight paths:

```bash
python CNN_autoencoder.py
python multi_view_classifier.py
```

The first script reports the shapes of the processed dataset, training/test splits, and group-mean targets. The second script creates the classifier dataset and runs training, reporting Keras training and validation metrics. These commands **require the external data described above** and have not been validated as a self-contained run from a fresh clone.

## Reproducibility and scope

This repository documents the **autoencoder and classifier components** of the thesis analysis. It currently provides model definitions, preprocessing steps, classifier training code, and two saved autoencoder weight files.

It does **not** currently include the raw cavity recordings, signal-generation/injection scripts, a complete autoencoder training procedure, or the matched-filtering and excess-power benchmark implementations discussed in the wider thesis work. Accordingly, the repository alone is **not a complete end-to-end reproduction** of all thesis results.

For the broader scientific motivation, methods, and results, see the [thesis PDF](Saraf_Swarangee_Master_s_Thesis.pdf).

## Author

**Swarangee Saraf**  
M.Sc. Astrophysics, University of Bonn
