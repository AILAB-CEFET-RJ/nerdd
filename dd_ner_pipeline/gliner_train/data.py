from torch.utils.data import DataLoader

from gliner_train.io_utils import load_jsonl


def tokenize_with_spans(text):
    """Split text by whitespace and return (tokens, char spans)."""
    tokens = []
    token_spans = []
    start = 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            break
        end = start
        while end < len(text) and not text[end].isspace():
            end += 1
        token = text[start:end]
        tokens.append(token)
        token_spans.append((start, end))
        start = end
    return tokens, token_spans


def process_sample(sample):
    """Map char spans to token spans for a single example."""
    text = sample["text"]
    spans = sample.get("spans", [])
    tokens, token_spans = tokenize_with_spans(text)

    ner = []
    for span in spans:
        start_char = span["start"]
        end_char = span["end"]
        label = span["label"]
        start_token = None
        end_token = None

        for i, (t_start, t_end) in enumerate(token_spans):
            if t_start >= start_char and t_end <= end_char:
                if start_token is None:
                    start_token = i
                end_token = i
            elif t_start < end_char and t_end > start_char:
                if start_token is None:
                    start_token = i
                end_token = i

        if start_token is not None and end_token is not None:
            ner.append([start_token, end_token, label])

    sample_id = sample.get("sample_id")
    return {"tokenized_text": tokens, "ner": ner, "sample_id": sample_id}


def load_dataset(path):
    """Load JSONL and preprocess into tokenized samples."""
    dataset = []
    for index, sample in enumerate(load_jsonl(path)):
        enriched = dict(sample)
        enriched.setdefault("sample_id", f"sample_{index}")
        dataset.append(process_sample(enriched))
    return dataset


def split_long_sentences(dataset, max_length=384, overlap=50):
    """Split long tokenized sequences with overlap, adjusting entity spans."""
    split_data = []
    for sample in dataset:
        words = sample.get("tokenized_text", [])
        ner_annotations = sample.get("ner", [])

        if not words or not isinstance(ner_annotations, list):
            continue

        if len(words) > max_length:
            step = max_length - overlap
            if step <= 0:
                step = max_length // 2 if max_length > 1 else 1

            for i in range(0, len(words), step):
                segment_start_idx = i
                segment_end_idx = min(i + max_length, len(words))

                if i > 0:
                    segment_start_idx = max(0, i - overlap)

                current_words = words[segment_start_idx:segment_end_idx]

                new_ner = []
                for start, end, label in ner_annotations:
                    if max(start, segment_start_idx) <= min(end, segment_end_idx - 1):
                        adjusted_start = max(0, start - segment_start_idx)
                        adjusted_end = min(len(current_words) - 1, end - segment_start_idx)
                        if adjusted_start <= adjusted_end:
                            new_ner.append([adjusted_start, adjusted_end, label])

                split_sample = {
                    "tokenized_text": current_words,
                    "ner": new_ner,
                    "sample_id": sample.get("sample_id"),
                }
                if split_sample["ner"]:
                    split_data.append(split_sample)
        else:
            if ner_annotations:
                split_data.append(
                    {
                        "tokenized_text": words,
                        "ner": ner_annotations,
                        "sample_id": sample.get("sample_id"),
                    }
                )

    return [ex for ex in split_data if "ner" in ex and isinstance(ex["ner"], list) and ex["ner"]]


def create_dataloader(dataset, batch_size, collator, shuffle=True):
    """Build a torch DataLoader with the GLiNER collator."""
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=shuffle)


def token_spans_to_char_offsets(tokens, spans):
    """Convert token index spans back into char spans over a joined text."""
    char_spans = []
    text = " ".join(tokens)
    current_pos = 0
    token_offsets = []

    for token in tokens:
        start = text.find(token, current_pos)
        end = start + len(token)
        token_offsets.append((start, end))
        current_pos = end + 1

    for start_idx, end_idx, label in spans:
        if start_idx < len(token_offsets) and end_idx < len(token_offsets):
            char_start = token_offsets[start_idx][0]
            char_end = token_offsets[end_idx][1]
            char_spans.append({"start": char_start, "end": char_end, "label": label})

    return text, char_spans
