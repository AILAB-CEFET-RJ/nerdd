import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return _FakeTensor(self.value)


sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        no_grad=lambda: _NoGrad(),
        nn=types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=lambda *args, **kwargs: None)),
    ),
)
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda iterable, **kwargs: iterable))

from base_model_training.engine import train_with_early_stopping


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _Loss:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value

    def backward(self):
        return None


class _ModelOutput:
    def __init__(self, loss):
        self.loss = _Loss(loss)


class _FakeModel:
    def __init__(self):
        self.weight = 0
        self.device = None

    def train(self):
        return None

    def eval(self):
        return None

    def __call__(self, **batch):
        return _ModelOutput(1.0)

    def parameters(self):
        return []

    def state_dict(self):
        return {"weight": _FakeTensor(self.weight)}

    def load_state_dict(self, state_dict):
        self.weight = state_dict["weight"].value

    def to(self, device):
        self.device = device
        return self


class _FakeValue:
    def to(self, device):
        return self


class _FakeOptimizer:
    def zero_grad(self):
        return None

    def step(self):
        return None


class BaseModelTrainingEngineTests(unittest.TestCase):
    def test_train_with_early_stopping_restores_best_checkpoint(self):
        model = _FakeModel()
        train_loader = [{"x": _FakeValue()}]
        val_loader = [{"x": _FakeValue()}]
        optimizer = _FakeOptimizer()
        metrics = iter([0.8, 0.7, 0.6])
        weights = iter([10, 20, 30])

        def metric_fn(current_model):
            current_model.weight = next(weights)
            return next(metrics)

        history = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            device="cpu",
            num_epochs=3,
            patience=2,
            metric_fn=metric_fn,
            stage_label="test",
        )

        self.assertEqual(history["best_metric"], 0.8)
        self.assertEqual(history["best_epoch"], 1)
        self.assertEqual(model.weight, 10)


if __name__ == "__main__":
    unittest.main()
