"""Utilities for loading JSON experiment configurations."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


METADATA_KEYS = {"experiment_id", "experiment_obs", "entrypoint", "n_repeats", "seed_start"}


def load_experiment_config_file(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON experiment config file as a list of dictionaries."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Experiment config must be a JSON object or a list of objects.")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("Experiment config list must contain only objects.")
    return payload


def select_experiment(
    experiments: list[dict[str, Any]],
    *,
    experiment_id: str = "",
) -> dict[str, Any]:
    """Select one experiment from a loaded config list."""
    if not experiments:
        raise ValueError("Experiment config is empty.")
    if experiment_id:
        matches = [item for item in experiments if item.get("experiment_id") == experiment_id]
        if not matches:
            raise ValueError(f"Experiment id not found: {experiment_id}")
        if len(matches) > 1:
            raise ValueError(f"Experiment id is duplicated: {experiment_id}")
        return dict(matches[0])
    if len(experiments) > 1:
        raise ValueError("Config has multiple experiments; provide --experiment-id or use --all.")
    return dict(experiments[0])


def dataclass_field_names(config_class: type) -> set[str]:
    if not is_dataclass(config_class):
        raise TypeError(f"Expected a dataclass type, got {config_class!r}")
    return {field.name for field in fields(config_class)}


def split_metadata_and_params(
    raw: dict[str, Any],
    *,
    config_class: type,
    required_entrypoint: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split experiment metadata from dataclass constructor parameters."""
    allowed = dataclass_field_names(config_class)
    metadata = {key: raw[key] for key in METADATA_KEYS if key in raw}
    entrypoint = metadata.get("entrypoint")
    if required_entrypoint is not None and entrypoint not in (None, required_entrypoint):
        raise ValueError(f"Unsupported entrypoint for this command: {entrypoint}")

    unknown = sorted(set(raw) - allowed - METADATA_KEYS)
    if unknown:
        raise ValueError(f"Unknown config key(s): {', '.join(unknown)}")
    params = {key: value for key, value in raw.items() if key in allowed}
    return metadata, params


def normalize_thresholds(value: Any) -> list[float]:
    if value is None:
        return value
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        if not pieces:
            raise ValueError("thresholds must not be empty.")
        return [float(piece) for piece in pieces]
    if isinstance(value, list):
        if not value:
            raise ValueError("thresholds must not be empty.")
        return [float(item) for item in value]
    raise ValueError("thresholds must be a string or a list of numbers.")


def build_dataclass_config_from_experiment(
    raw: dict[str, Any],
    *,
    config_class: type,
    required_entrypoint: str | None = None,
    defaults: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create a dataclass config instance from one experiment object."""
    metadata, params = split_metadata_and_params(
        raw,
        config_class=config_class,
        required_entrypoint=required_entrypoint,
    )
    if "thresholds" in params:
        params["thresholds"] = normalize_thresholds(params["thresholds"])
    base = defaults if defaults is not None else config_class()
    values = {field.name: getattr(base, field.name) for field in fields(config_class)}
    values.update(params)
    return config_class(**values), metadata


def load_dataclass_config(
    path: str | Path,
    *,
    config_class: type,
    experiment_id: str = "",
    required_entrypoint: str | None = None,
    defaults: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    experiments = load_experiment_config_file(path)
    selected = select_experiment(experiments, experiment_id=experiment_id)
    return build_dataclass_config_from_experiment(
        selected,
        config_class=config_class,
        required_entrypoint=required_entrypoint,
        defaults=defaults,
    )
