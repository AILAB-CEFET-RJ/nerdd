import json


def load_jsonl(path):
    """Load JSONL file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def save_jsonl(path, entries):
    """Save a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
