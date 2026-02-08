from pathlib import Path


def resolve_path(base_dir, path):
    """Resolve relative paths from a base directory."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate
