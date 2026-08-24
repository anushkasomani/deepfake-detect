from __future__ import annotations

import os
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.audio_features import extract_features, load_audio
from src.model_utils import load_artifact

st.set_page_config(page_title='Deepfake Audio Detection', page_icon='🎙️', layout='wide')
st.title('Deepfake Audio Detection')
st.caption('Upload an audio file to classify it as Genuine or Deepfake.')

MODEL_PATH = Path('artifacts/model.joblib')


@st.cache_resource
def load_model():
    return load_artifact(MODEL_PATH) if MODEL_PATH.exists() else None


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or '.wav'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 2.7))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title('Waveform')
    fig.tight_layout()
    return fig


def plot_mel_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title('Mel Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    fig.tight_layout()
    return fig


model = load_model()
if model is None:
    st.warning('No trained model found. Run `python train.py` first to create `artifacts/model.joblib`.')

uploaded = st.file_uploader('Choose an audio file', type=['wav', 'mp3', 'flac', 'm4a', 'ogg', 'aac', 'wma'])

if uploaded is not None:
    tmp_path = save_uploaded_file(uploaded)
    try:
        y, sr = load_audio(tmp_path)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader('Preview')
            st.audio(tmp_path)
            st.write(f'Sample rate: **{sr} Hz**')
            st.write(f'Duration: **{len(y)/sr:.2f} s**')
        with c2:
            st.subheader('Visuals')
            st.pyplot(plot_waveform(y, sr))
            st.pyplot(plot_mel_spectrogram(y, sr))

        if model is not None:
            feat = extract_features(tmp_path)
            proba = model.predict_proba([feat])[0]
            pred = int(np.argmax(proba))
            label = 'Deepfake (AI-Generated)' if pred == 1 else 'Genuine (Human)'
            st.divider()
            st.subheader('Prediction')
            st.metric('Class', label)
            st.metric('Confidence', f'{proba[pred]*100:.2f}%')
            st.write({'genuine_probability': float(proba[0]), 'deepfake_probability': float(proba[1])})
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

with st.expander('About'):
    st.markdown(
        'Feature-based classifier (MFCC, mel, spectral stats) trained on the '
        '[Fake-or-Real for-norm dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset).'
    )