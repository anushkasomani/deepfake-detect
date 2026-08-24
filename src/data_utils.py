from __future__ import annotations

from pathlib import Path

import pandas as pd

from .audio_features import SUPPORTED_AUDIO_EXTENSIONS

GENUINE_ALIASES = frozenset({'genuine', 'real', 'bonafide', 'human', 'original'})
DEEPFAKE_ALIASES = frozenset({'deepfake', 'fake', 'spoof', 'synthetic', 'generated'})

FOR_NORM_TRAIN_SUFFIXES = (
    'for-norm/for-norm/training',
    'for-norm/for-norm/train',
    'for-norm/training',
    'for-norm/train',
)


def _has_class_folders(path: Path) -> bool:
    if not path.is_dir():
        return False
    child_names = {child.name.lower() for child in path.iterdir() if child.is_dir()}
    return bool(child_names & GENUINE_ALIASES and child_names & DEEPFAKE_ALIASES)


def resolve_train_dir(data_root: str | Path) -> Path:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f'Data root does not exist: {root}')

    if _has_class_folders(root):
        return root

    for suffix in FOR_NORM_TRAIN_SUFFIXES:
        candidate = root / suffix
        if _has_class_folders(candidate):
            return candidate

    for candidate in root.rglob('*'):
        if candidate.is_dir() and _has_class_folders(candidate):
            return candidate

    raise FileNotFoundError(
        f'Could not find a train split under {root}. '
        'Expected folders like real/ and fake/.'
    )


def resolve_kaggle_dataset_root() -> Path:
    for root in (
        '/kaggle/input/the-fake-or-real-dataset',
        '/kaggle/input/for-norm',
    ):
        root_path = Path(root)
        if not root_path.exists():
            continue
        try:
            return resolve_train_dir(root_path)
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        'Kaggle dataset not found. Add mohammedabdeldayem/the-fake-or-real-dataset via Add Data.'
    )


def infer_label_map(data_dir: str | Path) -> dict[str, int]:
    data_dir = Path(data_dir)
    label_map: dict[str, int] = {}
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name.lower()
        if name in GENUINE_ALIASES:
            label_map[child.name] = 0
        elif name in DEEPFAKE_ALIASES:
            label_map[child.name] = 1
    if len(label_map) < 2:
        raise FileNotFoundError(f'Could not infer class folders under {data_dir}.')
    return label_map


def scan_audio_files(data_dir: str | Path, label_map: dict[str, int]) -> pd.DataFrame:
    data_dir = Path(data_dir)
    rows = []
    for class_name, label in label_map.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for path in class_dir.rglob('*'):
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                rows.append({'path': str(path), 'label': label, 'class_name': class_name})
    if not rows:
        raise FileNotFoundError(f'No audio files found in {data_dir}')
    return pd.DataFrame(rows)
