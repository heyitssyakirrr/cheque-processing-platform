"""Load a fine-tuned digit classifier and classify canonical digit images
produced by digit_segmentation.py.

Swap models by changing `digit_model_backend` / `digit_model_path` in
settings (or the equivalent .env values) -- no other code changes needed.

Supported backends:
- "transformers"  -> any AutoModelForImageClassification repo, e.g.
                     farleyknight/mnist-digit-classification-2022-09-04
- "torch_pickle"  -> a plain torch.save(model) checkpoint trained on
                     28x28 MNIST-normalized input, e.g.
                     developerPratik/mnist-cnn-classifier (best_model.pth)
"""

import threading
from typing import Any

import numpy as np
import torch
from PIL import Image

from app.settings import settings

_model: Any | None = None
_processor: Any | None = None
_lock = threading.Lock()

_MNIST_MEAN, _MNIST_STD = 0.1307, 0.3081


def _load() -> None:
    global _model, _processor
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        if settings.digit_model_backend == "transformers":
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            _processor = AutoImageProcessor.from_pretrained(settings.digit_model_path)
            _model = AutoModelForImageClassification.from_pretrained(settings.digit_model_path)
            _model.eval()
        elif settings.digit_model_backend == "torch_pickle":
            # weights_only=False: this checkpoint is a full pickled nn.Module,
            # not just a state_dict. Only safe because the file has already
            # been downloaded and vetted locally -- never point this at a
            # live/untrusted URL.
            _model = torch.load(settings.digit_model_path, map_location="cpu", weights_only=False)
            _model.eval()
        else:
            raise ValueError(f"Unknown digit_model_backend: {settings.digit_model_backend}")


def _label_to_digit(index: int) -> str:
    id2label = getattr(_model.config, "id2label", None) if hasattr(_model, "config") else None
    if not id2label:
        return str(index)
    label = str(id2label.get(index, index))
    return label.split("_")[-1] if label.upper().startswith("LABEL_") else label


def _predict_transformers(canonical: np.ndarray) -> tuple[str, float]:
    image = Image.fromarray(canonical).convert("RGB")
    inputs = _processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        probs = torch.softmax(_model(**inputs).logits, dim=1).squeeze()
    index = int(torch.argmax(probs).item())
    return _label_to_digit(index), float(probs[index].item())


def _predict_torch_pickle(canonical: np.ndarray) -> tuple[str, float]:
    resized = np.array(Image.fromarray(canonical).resize((28, 28)))
    tensor = torch.from_numpy(resized).float().div(255)
    tensor = (tensor - _MNIST_MEAN) / _MNIST_STD
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    with torch.inference_mode():
        probs = torch.softmax(_model(tensor), dim=1).squeeze()
    index = int(torch.argmax(probs).item())
    return str(index), float(probs[index].item())


def classify_digits(canonical_slices: list[np.ndarray]) -> list[tuple[str, float]]:
    """Return (digit_char, confidence) for each canonical digit image, in order."""
    _load()
    predict = _predict_transformers if settings.digit_model_backend == "transformers" else _predict_torch_pickle
    return [predict(digit_image) for digit_image in canonical_slices]