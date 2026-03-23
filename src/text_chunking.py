def special_tokens_to_add(tokenizer):
    if tokenizer is None or not hasattr(tokenizer, "num_special_tokens_to_add"):
        return 0
    try:
        return int(tokenizer.num_special_tokens_to_add(pair=False))
    except TypeError:
        return int(tokenizer.num_special_tokens_to_add())


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


def split_text_fast(text, model, tokenizer, max_tokens):
    budget = effective_chunk_budget(model, tokenizer, max_tokens)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for index in range(0, len(token_ids), budget):
        chunk_ids = token_ids[index : index + budget]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks
