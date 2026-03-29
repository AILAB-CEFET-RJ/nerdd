from pathlib import Path


def resolve_path(_base_dir, path):
    """Resolve paths using CLI semantics.

    Absolute paths are returned as-is. Relative paths are resolved from the
    current working directory, which matches how users invoke commands.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def resolve_repo_artifact_path(anchor_file, path):
    """Resolve artifacts paths from repo root while preserving CLI semantics elsewhere."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    raw = str(path)
    if raw == "artifacts" or raw.startswith("artifacts/") or raw == "./artifacts" or raw.startswith("./artifacts/"):
        repo_root = Path(anchor_file).resolve().parents[2]
        normalized = raw[2:] if raw.startswith("./") else raw
        return repo_root / normalized

    return Path.cwd() / candidate
