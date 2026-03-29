from collections import defaultdict


DEFAULT_LABEL_MAP = {
    "person": "Person",
    "location": "Location",
    "organization": "Organization",
}


def normalize_gliner2_entities(entities, label_map=None):
    label_map = dict(DEFAULT_LABEL_MAP if label_map is None else label_map)
    normalized = []

    if isinstance(entities, dict):
        flattened = []
        for label, values in entities.items():
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, dict):
                        flattened.append({"label": label, **value})
                    else:
                        flattened.append({"label": label, "text": str(value)})
            elif isinstance(values, dict):
                flattened.append({"label": label, **values})
            else:
                flattened.append({"label": label, "text": str(values)})
        entities = flattened

    if not isinstance(entities, list):
        return []

    for entity in entities:
        if isinstance(entity, str):
            entity = {"text": entity}
        if not isinstance(entity, dict):
            continue
        raw_label = str(entity.get("label", "")).strip()
        mapped_label = label_map.get(raw_label.lower(), raw_label)
        normalized_entity = {
            "text": entity.get("text", ""),
            "label": mapped_label,
        }
        if entity.get("start") is not None:
            normalized_entity["start"] = int(entity["start"])
        if entity.get("end") is not None:
            normalized_entity["end"] = int(entity["end"])
        if entity.get("confidence") is not None:
            normalized_entity["score"] = float(entity["confidence"])
        elif entity.get("score") is not None:
            normalized_entity["score"] = float(entity["score"])
        normalized.append(normalized_entity)

    normalized.sort(
        key=lambda entity: (
            str(entity.get("label", "")),
            int(entity.get("start", -1)),
            int(entity.get("end", -1)),
            str(entity.get("text", "")),
        )
    )
    return normalized


def _find_all_occurrences(text, needle):
    if not needle:
        return []
    positions = []
    start = 0
    while True:
        pos = text.find(needle, start)
        if pos == -1:
            break
        positions.append((pos, pos + len(needle)))
        start = pos + 1
    return positions


def attach_missing_offsets(text, entities):
    occupied = set()
    completed = []
    by_text = defaultdict(list)

    for entity in entities:
        if entity.get("start") is not None and entity.get("end") is not None:
            key = (entity["start"], entity["end"], entity.get("label", ""), entity.get("text", ""))
            occupied.add((entity["start"], entity["end"], entity.get("text", "")))
            completed.append(entity)
        else:
            by_text[str(entity.get("text", ""))].append(entity)

    for entity_text, text_entities in by_text.items():
        occurrences = _find_all_occurrences(text, entity_text)
        occurrence_index = 0
        for entity in text_entities:
            fixed = dict(entity)
            while occurrence_index < len(occurrences):
                start, end = occurrences[occurrence_index]
                occurrence_index += 1
                if (start, end, entity_text) in occupied:
                    continue
                fixed["start"] = start
                fixed["end"] = end
                occupied.add((start, end, entity_text))
                break
            completed.append(fixed)

    completed.sort(
        key=lambda entity: (
            int(entity.get("start", -1)),
            int(entity.get("end", -1)),
            str(entity.get("label", "")),
            str(entity.get("text", "")),
        )
    )
    return completed


def deduplicate_entities(entities):
    seen = set()
    deduped = []
    for entity in entities:
        key = (
            entity.get("start"),
            entity.get("end"),
            entity.get("label"),
            entity.get("text", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def predict_entities_for_text(model, text, entity_types, label_map=None):
    raw = model.extract_entities(text, entity_types)
    if isinstance(raw, dict):
        raw = raw.get("entities", raw)
    normalized = normalize_gliner2_entities(raw, label_map=label_map)
    with_offsets = attach_missing_offsets(text, normalized)
    valid = []
    for entity in with_offsets:
        start = entity.get("start")
        end = entity.get("end")
        if start is None or end is None:
            continue
        if start < 0 or end > len(text) or start >= end:
            continue
        valid.append(entity)
    return deduplicate_entities(valid)
