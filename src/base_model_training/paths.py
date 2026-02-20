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
