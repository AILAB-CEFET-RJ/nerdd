import json


def load_jsonl(path):
    """Load records from JSONL, JSON array, or a single JSON object."""
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON payload in {path}: expected object, array, or JSONL")


def save_jsonl(path, entries):
    """Save a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
