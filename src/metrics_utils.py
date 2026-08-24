from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def equal_error_rate(y_true, y_score) -> float:
    """Compute EER from binary labels and deepfake (positive-class) scores."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0:
        raise ValueError('Cannot compute EER on an empty set.')
    if len(np.unique(y_true)) < 2:
        raise ValueError('EER requires both classes in y_true.')

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)
