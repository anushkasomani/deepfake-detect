from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.audio_features import FeatureConfig, extract_features
from src.data_utils import infer_label_map, resolve_kaggle_dataset_root, resolve_train_dir, scan_audio_files
from src.metrics_utils import equal_error_rate
from src.model_utils import save_artifact


def parse_args():
    p = argparse.ArgumentParser(description='Train deepfake audio detector')
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--output_dir', type=str, default='artifacts')
    p.add_argument('--genuine_folder', type=str, default=None)
    p.add_argument('--deepfake_folder', type=str, default=None)
    p.add_argument('--kaggle', action='store_true', help='Auto-locate attached Kaggle dataset')
    p.add_argument('--max_samples', type=int, default=None, help='Cap files per class for quick runs')
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--random_state', type=int, default=42)
    return p.parse_args()


def resolve_data_dir(args) -> tuple[Path, dict[str, int]]:
    if args.kaggle:
        train_dir = resolve_kaggle_dataset_root()
        label_map = infer_label_map(train_dir)
        print(f'Using Kaggle train dir: {train_dir}')
        return train_dir, label_map

    data_dir = resolve_train_dir(args.data_dir)
    if args.genuine_folder and args.deepfake_folder:
        label_map = {args.genuine_folder: 0, args.deepfake_folder: 1}
    else:
        label_map = infer_label_map(data_dir)
    print(f'Using train dir: {data_dir}')
    return data_dir, label_map


def maybe_subsample(df, max_samples: int | None, random_state: int):
    if max_samples is None:
        return df
    parts = []
    for label in sorted(df['label'].unique()):
        subset = df[df['label'] == label]
        parts.append(subset.sample(n=min(max_samples, len(subset)), random_state=random_state))
    return pd.concat(parts, ignore_index=True)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_dir, label_map = resolve_data_dir(args)
    df = scan_audio_files(data_dir, label_map)
    df = maybe_subsample(df, args.max_samples, args.random_state)
    print(f'Found {len(df)} files')
    print(f'Label map: {label_map}')

    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting features'):
        try:
            X.append(extract_features(row['path'], FeatureConfig()))
            y.append(row['label'])
        except Exception as exc:
            print(f'[WARN] Skipping {row["path"]}: {exc}')

    X = np.vstack(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=400,
            class_weight='balanced',
            random_state=args.random_state,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        'accuracy': float(accuracy_score(y_val, pred)),
        'macro_f1': float(f1_score(y_val, pred, average='macro')),
        'eer': float(equal_error_rate(y_val, proba)),
        'confusion_matrix': confusion_matrix(y_val, pred).tolist(),
        'classification_report': classification_report(y_val, pred, output_dict=True),
        'feature_dim': int(X.shape[1]),
        'train_size': int(len(X_train)),
        'val_size': int(len(X_val)),
        'data_dir': str(data_dir),
        'label_map': label_map,
    }

    save_artifact(model, out / 'model.joblib')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f'Saved model to {out / "model.joblib"}')


if __name__ == '__main__':
    main()
