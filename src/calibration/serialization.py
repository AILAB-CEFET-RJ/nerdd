import json
from pathlib import Path

import numpy as np

try:
    from calibration.methods.temperature import apply_temperature
except ImportError:  # pragma: no cover
    from methods.temperature import apply_temperature


def _clip_score(score):
    return float(min(max(float(score), 1e-6), 1.0 - 1e-6))


def save_calibrator(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_calibrator(path):
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_calibrator_to_score(score, label, calibrator):
    method = calibrator["method"]
    score = _clip_score(score)

    if method == "temperature":
        temperature = float(calibrator["parameters"]["temperature"])
        return float(apply_temperature(np.asarray([score], dtype=np.float64), temperature)[0])

    if method == "temperature-per-class":
        class_params = calibrator["parameters"]["per_class_temperature"]
        fallback = calibrator["parameters"].get("global_fallback_temperature")
        label_params = class_params.get(label)
        temperature = None
        if label_params and label_params.get("temperature") is not None:
            temperature = float(label_params["temperature"])
        elif fallback is not None:
            temperature = float(fallback)
        if temperature is None:
            return score
        return float(apply_temperature(np.asarray([score], dtype=np.float64), temperature)[0])

    if method == "isotonic":
        iso_x = np.asarray(calibrator["parameters"]["isotonic_x_thresholds"], dtype=np.float64)
        iso_y = np.asarray(calibrator["parameters"]["isotonic_y_thresholds"], dtype=np.float64)
        return float(np.interp(np.asarray([score], dtype=np.float64), iso_x, iso_y)[0])

    raise ValueError(f"Unsupported calibrator method: {method}")
