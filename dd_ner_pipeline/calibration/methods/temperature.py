import numpy as np


EPS = 1e-6


def _to_logits(probs):
    clipped = np.clip(probs, EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _bce(y_true, y_prob):
    y_prob = np.clip(y_prob, EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def fit_temperature(scores, labels, t_min=0.5, t_max=5.0, grid_size=181):
    if len(scores) == 0:
        raise ValueError("No scores available for temperature fitting.")
    if len(set(labels.tolist())) < 2:
        raise ValueError("Temperature fitting requires both positive and negative labels.")

    logits = _to_logits(scores)
    grid = np.linspace(t_min, t_max, grid_size)
    losses = []
    for t in grid:
        calibrated = _sigmoid(logits / t)
        losses.append(_bce(labels, calibrated))

    best_idx = int(np.argmin(losses))
    return float(grid[best_idx]), float(losses[best_idx])


def apply_temperature(scores, temperature):
    logits = _to_logits(scores)
    calibrated = _sigmoid(logits / temperature)
    return calibrated
