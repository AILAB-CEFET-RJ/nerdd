import logging


def build_inference_gliner_kwargs(model_max_length=0, load_tokenizer=True):
    kwargs = {}
    if load_tokenizer:
        kwargs["load_tokenizer"] = True
    if model_max_length and int(model_max_length) > 0:
        kwargs["max_length"] = int(model_max_length)
    return kwargs


def load_gliner_model(model_path, *, model_max_length=0, load_tokenizer=True, logger=None, context=""):
    from gliner import GLiNER

    kwargs = build_inference_gliner_kwargs(
        model_max_length=model_max_length,
        load_tokenizer=load_tokenizer,
    )

    if logger is None:
        logger = logging.getLogger(__name__)
    prefix = f"{context} " if context else ""
    logger.info("Loading %sGLiNER model from: %s", prefix, model_path)
    if "max_length" in kwargs:
        logger.info("Using %sGLiNER model max_length=%s", prefix, kwargs["max_length"])

    return GLiNER.from_pretrained(model_path, **kwargs)
