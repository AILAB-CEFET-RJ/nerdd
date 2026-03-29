#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from gliner2 import GLiNER2

try:
    from gliner2.training.lora import LoRALayer
except Exception:  # pragma: no cover - optional runtime import
    LoRALayer = None


DEFAULT_MODEL = "fastino/gliner2-base-v1"
DEFAULT_ADAPTER_DIR = "outputs/nerdd_lora/final"
DEFAULT_ENTITY_TYPES = ["person", "location", "organization"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the trained NERDD LoRA adapter and compare against the base model."
    )
    parser.add_argument(
        "--adapter-dir",
        default=DEFAULT_ADAPTER_DIR,
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Base GLiNER2 model name or local path.",
    )
    parser.add_argument(
        "--text",
        help="Single input text to evaluate.",
    )
    parser.add_argument(
        "--file",
        help="Path to a text file with one input per line.",
    )
    parser.add_argument(
        "--data",
        default="nerdd/dd_corpus_small_train.jsonl",
        help="Dataset used to sample examples when neither --text nor --file is provided.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="Number of dataset samples to use when neither --text nor --file is provided.",
    )
    parser.add_argument(
        "--compare-base",
        dest="compare_base",
        action="store_true",
        default=True,
        help="Compare adapter predictions against the base model.",
    )
    parser.add_argument(
        "--no-compare-base",
        dest="compare_base",
        action="store_false",
        help="Disable comparison against the base model.",
    )
    return parser.parse_args()


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.text:
        return [args.text.strip()]

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        texts = [
            line.strip()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not texts:
            raise ValueError(f"No non-empty lines found in {file_path}")
        return texts

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Sample dataset not found: {data_path}")

    texts: list[str] = []
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("input", "").strip()
            if text:
                texts.append(text)
            if len(texts) >= args.sample_count:
                break

    if not texts:
        raise ValueError(f"No usable samples found in {data_path}")

    return texts


def count_lora_layers(model: GLiNER2) -> int | None:
    if LoRALayer is None:
        return None
    return sum(1 for module in model.modules() if isinstance(module, LoRALayer))


def extract_entities(model: GLiNER2, text: str) -> object:
    result = model.extract_entities(text, DEFAULT_ENTITY_TYPES)
    if isinstance(result, dict):
        return result.get("entities", result)
    return result


def normalize_entities(entities: object) -> list[dict]:
    if isinstance(entities, list):
        normalized: list[dict] = []
        for entity in entities:
            if isinstance(entity, dict):
                normalized.append(entity)
            elif isinstance(entity, str):
                normalized.append({"text": entity})
        return sort_entities(normalized)

    if isinstance(entities, dict):
        normalized = []
        for label, mentions in entities.items():
            if isinstance(mentions, list):
                for mention in mentions:
                    if isinstance(mention, dict):
                        normalized.append({"label": label, **mention})
                    else:
                        normalized.append({"text": str(mention), "label": label})
            elif isinstance(mentions, dict):
                normalized.append({"label": label, **mentions})
            else:
                normalized.append({"text": str(mentions), "label": label})
        return sort_entities(normalized)

    return []


def sort_entities(entities: list[dict]) -> list[dict]:
    return sorted(
        entities,
        key=lambda entity: (
            str(entity.get("label", "")),
            str(entity.get("text", "")),
            str(entity.get("start", "")),
            str(entity.get("end", "")),
        ),
    )


def format_entity(entity: dict) -> str:
    text = entity.get("text", "<missing>")
    label = entity.get("label", "<missing>")
    confidence = entity.get("confidence")
    start = entity.get("start")
    end = entity.get("end")

    parts = [f"text={text!r}", f"label={label!r}"]
    if confidence is not None:
        parts.append(f"confidence={confidence:.4f}")
    if start is not None and end is not None:
        parts.append(f"span=({start}, {end})")
    return ", ".join(parts)


def print_entities(title: str, entities: list[dict]) -> None:
    entities = normalize_entities(entities)
    print(title)
    if not entities:
        print("  (no entities)")
        return
    for entity in entities:
        print(f"  - {format_entity(entity)}")


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    texts = load_texts(args)

    print("Loading base model...")
    model = GLiNER2.from_pretrained(args.model)

    print(f"Loading adapter from {adapter_dir}...")
    model.load_adapter(str(adapter_dir))
    print(f"Has adapter: {getattr(model, 'has_adapter', 'unknown')}")

    lora_count = count_lora_layers(model)
    if lora_count is None:
        print("LoRA layers: unavailable")
    else:
        print(f"LoRA layers: {lora_count}")

    print(f"Entity types: {DEFAULT_ENTITY_TYPES}")
    print(f"Texts to evaluate: {len(texts)}")

    for idx, text in enumerate(texts, start=1):
        print("\n" + "=" * 80)
        print(f"Sample {idx}")
        print("=" * 80)
        print(text)

        adapter_entities = extract_entities(model, text)
        print_entities("Adapter output:", adapter_entities)

        if args.compare_base:
            model.unload_adapter()
            print(f"Has adapter after unload: {getattr(model, 'has_adapter', 'unknown')}")
            base_entities = extract_entities(model, text)
            print_entities("Base model output:", base_entities)

            model.load_adapter(str(adapter_dir))
            print(f"Has adapter after reload: {getattr(model, 'has_adapter', 'unknown')}")


if __name__ == "__main__":
    main()
