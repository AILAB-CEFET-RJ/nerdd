import logging


def load_gliner2_model(
    model_path,
    *,
    adapter_dir="",
    logger=None,
    context="",
):
    from gliner2 import GLiNER2

    if logger is None:
        logger = logging.getLogger(__name__)

    prefix = f"{context} " if context else ""
    logger.info("Loading %sGLiNER2 model from: %s", prefix, model_path)
    model = GLiNER2.from_pretrained(model_path)

    if adapter_dir:
        logger.info("Loading %sGLiNER2 adapter from: %s", prefix, adapter_dir)
        model.load_adapter(str(adapter_dir))

    return model
