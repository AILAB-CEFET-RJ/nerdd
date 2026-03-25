import logging


def build_inference_gliner_kwargs(model_max_length=0, load_tokenizer=True, map_location=""):
    kwargs = {}
    if load_tokenizer:
        kwargs["load_tokenizer"] = True
    if model_max_length and int(model_max_length) > 0:
        kwargs["max_length"] = int(model_max_length)
    if map_location:
        kwargs["map_location"] = str(map_location)
    return kwargs


def load_gliner_model(
    model_path,
    *,
    model_max_length=0,
    load_tokenizer=True,
    map_location="",
    logger=None,
    context="",
):
    from gliner import GLiNER

    kwargs = build_inference_gliner_kwargs(
        model_max_length=model_max_length,
        load_tokenizer=load_tokenizer,
        map_location=map_location,
    )

    if logger is None:
        logger = logging.getLogger(__name__)
    prefix = f"{context} " if context else ""
    logger.info("Loading %sGLiNER model from: %s", prefix, model_path)
    if "max_length" in kwargs:
        logger.info("Using %sGLiNER model max_length=%s", prefix, kwargs["max_length"])
    if "map_location" in kwargs:
        logger.info("Using %sGLiNER map_location=%s", prefix, kwargs["map_location"])

    return GLiNER.from_pretrained(model_path, **kwargs)
