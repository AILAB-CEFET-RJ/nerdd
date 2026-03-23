def tokenize_with_spans(text):
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


def special_tokens_to_add(tokenizer):
    if tokenizer is None or not hasattr(tokenizer, "num_special_tokens_to_add"):
        return 0
    try:
        return int(tokenizer.num_special_tokens_to_add(pair=False))
    except TypeError:
        return int(tokenizer.num_special_tokens_to_add())


def wordpiece_lengths(words, tokenizer):
    if tokenizer is None:
        return [1] * len(words)

    try:
        encoded = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        word_ids = encoded.word_ids()
        if word_ids is not None:
            lengths = [0] * len(words)
            for word_id in word_ids:
                if word_id is not None:
                    lengths[word_id] += 1
            return [length if length > 0 else 1 for length in lengths]
    except Exception:
        pass

    lengths = []
    for word in words:
        try:
            pieces = tokenizer.tokenize(word)
        except Exception:
            pieces = None
        lengths.append(len(pieces) if pieces else 1)
    return lengths


def find_chunk_end(wordpiece_lengths_values, start_idx, budget):
    used = 0
    end_idx = start_idx
    while end_idx < len(wordpiece_lengths_values):
        next_len = max(1, wordpiece_lengths_values[end_idx])
        if used and used + next_len > budget:
            break
        if not used and next_len > budget:
            return end_idx + 1
        used += next_len
        end_idx += 1
    return end_idx


def effective_chunk_budget(model, tokenizer, requested_max_tokens):
    candidates = [requested_max_tokens]
    processor_max_len = getattr(getattr(model, "data_processor", None), "max_len", None)
    if isinstance(processor_max_len, int) and processor_max_len > 0:
        candidates.append(processor_max_len)
    tokenizer_model_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_model_max, int) and 0 < tokenizer_model_max < 100000:
        candidates.append(tokenizer_model_max)
    effective_max = min(candidates)
    return max(1, effective_max - special_tokens_to_add(tokenizer))


def split_text_encoder_aware(text, model, tokenizer, max_tokens):
    words, word_spans = tokenize_with_spans(text)
    if not words:
        return []

    budget = effective_chunk_budget(model, tokenizer, max_tokens)
    wordpiece_lengths_values = wordpiece_lengths(words, tokenizer)
    chunks = []
    start_idx = 0
    while start_idx < len(words):
        end_idx = find_chunk_end(wordpiece_lengths_values, start_idx, budget)
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        start_char = word_spans[start_idx][0]
        end_char = word_spans[end_idx - 1][1]
        chunk_text = text[start_char:end_char].strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "start": start_char, "end": end_char})
        if end_idx >= len(words):
            break
        start_idx = end_idx
    return chunks
