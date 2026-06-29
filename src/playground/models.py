from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

BrowserEngine = Literal["wllama", "transformers"]


@dataclass(frozen=True)
class BrowserModel:
    id: str
    label: str
    engine: BrowserEngine
    repo: str
    file: str | None = None
    dtype: str | None = None
    device: str = "webgpu"

    def public_dict(self) -> dict[str, str]:
        return {key: value for key, value in asdict(self).items() if value is not None}


BROWSER_MODELS = (
    BrowserModel(
        id="bonsai-1.7b-q1",
        label="Bonsai 1.7B",
        engine="wllama",
        repo="prism-ml/Bonsai-1.7B-gguf",
        file="Bonsai-1.7B-Q1_0.gguf",
    ),
    BrowserModel(
        id="ternary-bonsai-1.7b-q2",
        label="Ternary Bonsai 1.7B",
        engine="transformers",
        repo="onnx-community/Ternary-Bonsai-1.7B-ONNX",
        dtype="q2",
    ),
)

MODEL_BY_ID = {model.id: model for model in BROWSER_MODELS}


def browser_model(model_id: str) -> BrowserModel:
    try:
        return MODEL_BY_ID[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown browser model: {model_id!r}") from exc


def public_model_registry() -> list[dict[str, str]]:
    return [model.public_dict() for model in BROWSER_MODELS]


__all__ = [
    "BROWSER_MODELS",
    "MODEL_BY_ID",
    "BrowserModel",
    "browser_model",
    "public_model_registry",
]
