import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeGLiNER:
    calls = []

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        cls.calls.append((model_path, kwargs))
        return {"model_path": model_path, "kwargs": kwargs}


sys.modules["gliner"] = types.SimpleNamespace(GLiNER=_FakeGLiNER)

from gliner_loader import build_inference_gliner_kwargs, load_gliner_model


class GLiNERLoaderTests(unittest.TestCase):
    def setUp(self):
        _FakeGLiNER.calls.clear()

    def test_build_kwargs_without_model_max_length(self):
        self.assertEqual(build_inference_gliner_kwargs(), {"load_tokenizer": True})

    def test_build_kwargs_with_model_max_length(self):
        self.assertEqual(
            build_inference_gliner_kwargs(model_max_length=384),
            {"load_tokenizer": True, "max_length": 384},
        )

    def test_build_kwargs_with_map_location(self):
        self.assertEqual(
            build_inference_gliner_kwargs(model_max_length=384, map_location="cuda"),
            {"load_tokenizer": True, "max_length": 384, "map_location": "cuda"},
        )

    def test_load_gliner_model_uses_shared_kwargs(self):
        result = load_gliner_model("model-path", model_max_length=384, map_location="cuda")
        self.assertEqual(result["model_path"], "model-path")
        self.assertEqual(
            _FakeGLiNER.calls,
            [("model-path", {"load_tokenizer": True, "max_length": 384, "map_location": "cuda"})],
        )


if __name__ == "__main__":
    unittest.main()
