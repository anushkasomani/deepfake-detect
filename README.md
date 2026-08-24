# Deepfake Audio Detection

Classifies speech recordings as **Genuine (Human)** or **Deepfake (AI-generated)**.

## Links

- **GitHub:** https://github.com/aara19/deepfake-audio-detection
- **Live demo:** [https://deepfake_audio.streamlit.app](https://deepfake-detect-123.streamlit.app)
- **Dataset:** [Fake-or-Real (for-norm)](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

## Problem

Generative AI can produce realistic synthetic speech. This project detects whether an audio clip is human or AI-generated.

## Methodology

### Preprocessing
- Load audio at 16 kHz mono (`librosa`)
- Trim silence, normalize amplitude
- Supported formats: wav, mp3, flac, m4a, ogg, aac, wma

### Feature extraction
Handcrafted features per clip:
- MFCCs + delta + delta-delta
- Mel spectrogram statistics
- Spectral: centroid, bandwidth, rolloff, flatness
- Chroma, zero-crossing rate, RMS, tempo, energy, duration

### Model
- `StandardScaler` + `RandomForestClassifier` (400 trees, balanced classes)
- Trained on a stratified subset of the for-norm split (500 samples per class)
- Labels: `0 = Genuine`, `1 = Deepfake`

### Metrics
Saved to `artifacts/metrics.json`:
- Accuracy
- Macro F1
- Equal Error Rate (EER)
- Confusion matrix
- Classification report

## Repository structure

```text
deepfake-audio-detection/
├── app.py                              # Streamlit web app
├── predict.py                          # CLI inference on a single file
├── train.py                            # Training script
├── requirements.txt
├── artifacts/                          # model.joblib + metrics.json (after training)
├── notebooks/
│   └── deepfake_audio_detection.ipynb  # Kaggle training notebook
└── src/
    ├── audio_features.py               # Preprocessing + feature extraction
    ├── data_utils.py                   # Dataset loading + Kaggle path resolution
    ├── metrics_utils.py                # EER calculation
    └── model_utils.py                  # Save/load model
```

## Train on Kaggle

The full dataset on Kaggle.

```

## Tech stack

Python · librosa · NumPy · scikit-learn · Joblib · Streamlit · Matplotlib
