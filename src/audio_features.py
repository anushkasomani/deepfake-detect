from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import librosa
import numpy as np

SUPPORTED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}

@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_mfcc: int = 20
    n_mels: int = 64
    max_seconds: Optional[float] = None


def _pad_or_trim(y: np.ndarray, target_length: int) -> np.ndarray:
    if len(y) > target_length:
        return y[:target_length]
    if len(y) < target_length:
        return np.pad(y, (0, target_length - len(y)))
    return y


def load_audio(file_path: str | Path, cfg: FeatureConfig = FeatureConfig()) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(file_path), sr=cfg.sample_rate, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if cfg.max_seconds is not None:
        y = _pad_or_trim(y, int(cfg.sample_rate * cfg.max_seconds))
    if y.size == 0:
        raise ValueError(f'Empty or unreadable audio: {file_path}')
    y = librosa.util.normalize(y)
    return y, sr


def _stats(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.array([np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x)], dtype=np.float32)


def extract_features(file_path: str | Path, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    y, sr = load_audio(file_path, cfg)

    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=cfg.n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.n_mfcc)
    mfcc_d1 = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if np.isnan(tempo):
        tempo = 0.0

    blocks: List[np.ndarray] = []
    for block in [zcr, rms, centroid, bandwidth, rolloff, flatness, chroma, mel_db, mfcc, mfcc_d1, mfcc_d2]:
        blocks.append(block.mean(axis=1).astype(np.float32))
        blocks.append(block.std(axis=1).astype(np.float32))

    duration = len(y) / sr
    energy = np.sum(y ** 2)

    blocks.append(np.array([float(tempo)], dtype=np.float32))
    blocks.append(_stats(y))
    blocks.append(np.array([float(energy), float(duration)], dtype=np.float32))

    return np.concatenate(blocks).astype(np.float32)