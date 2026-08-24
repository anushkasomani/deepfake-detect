from __future__ import annotations

import argparse
import json
import numpy as np

from src.audio_features import extract_features
from src.model_utils import load_artifact


def parse_args():
    p = argparse.ArgumentParser(description='Predict genuine vs deepfake for one audio file')
    p.add_argument('--audio', type=str, required=True)
    p.add_argument('--model', type=str, default='artifacts/model.joblib')
    return p.parse_args()


def main():
    args = parse_args()
    model = load_artifact(args.model)
    feat = extract_features(args.audio)
    proba = model.predict_proba([feat])[0]
    pred = int(np.argmax(proba))
    label = 'Deepfake (AI-Generated)' if pred == 1 else 'Genuine (Human)'
    result = {
        'audio': args.audio,
        'prediction': label,
        'confidence': float(proba[pred]),
        'probabilities': {
            'genuine': float(proba[0]),
            'deepfake': float(proba[1]),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()