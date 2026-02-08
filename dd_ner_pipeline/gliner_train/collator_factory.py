import inspect


def _candidate_kwargs(model):
    tokenizer = getattr(model.data_processor, "transformer_tokenizer", None)
    return [
        {"config": model.config, "data_processor": model.data_processor, "prepare_labels": True},
        {"data_processor": model.data_processor, "prepare_labels": True},
        {
            "config": model.config,
            "data_processor": model.data_processor,
            "tokenizer": tokenizer,
            "prepare_labels": True,
        },
        {"tokenizer": tokenizer, "prepare_labels": True},
        {},
    ]


def _filter_supported_kwargs(collator_cls, kwargs):
    signature = inspect.signature(collator_cls)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    accepted = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in accepted and value is not None}


def _pick_collator_class(collator_module, model):
    """Select collator class compatible with the loaded GLiNER model."""
    # Newer GLiNER versions expose architecture-specific collators.
    model_class_name = model.__class__.__name__  # e.g. UniEncoderSpanGLiNER
    dynamic_name = model_class_name.replace("GLiNER", "DataCollator")
    if hasattr(collator_module, dynamic_name):
        return getattr(collator_module, dynamic_name)

    # Backward compatibility with older versions.
    for legacy_name in ("DataCollator", "DataCollatorWithPadding"):
        if hasattr(collator_module, legacy_name):
            return getattr(collator_module, legacy_name)

    # Generic fallbacks by task style.
    if hasattr(collator_module, "SpanDataCollator"):
        return getattr(collator_module, "SpanDataCollator")
    if hasattr(collator_module, "TokenDataCollator"):
        return getattr(collator_module, "TokenDataCollator")
    return None


def build_data_collator(model):
    """Build a GLiNER data collator across package versions."""
    from gliner.data_processing import collator as collator_module

    collator_cls = _pick_collator_class(collator_module, model)

    if collator_cls is None:
        available = ", ".join(sorted(name for name in dir(collator_module) if "collator" in name.lower()))
        raise ImportError(
            "Could not find a compatible GLiNER collator class. "
            f"Available names in gliner.data_processing.collator: {available or 'none'}"
        )

    errors = []
    for kwargs in _candidate_kwargs(model):
        filtered = _filter_supported_kwargs(collator_cls, kwargs)
        try:
            return collator_cls(**filtered)
        except TypeError as exc:
            errors.append(f"kwargs={sorted(filtered.keys())}: {exc}")

    raise TypeError(
        f"Failed to initialize {collator_cls.__name__} with compatible arguments. Tried: "
        + " | ".join(errors)
    )
