from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

TorchTensor = torch.Tensor
TorchDevice = torch.device
TorchOptimizer = Optimizer


class VocabularyConfig:
    max_seq_length: int
    min_frequency: int
    specials: Sequence[str]

    def __init__(
        self,
        *,
        max_seq_length: int = ...,
        min_frequency: int = ...,
        specials: Sequence[str] | None = ...,
    ) -> None: ...


class EmotionTrainingConfig:
    enabled: bool


class MetaIntentStacker:
    def compute_adjustment(
        self, base_logits: TorchTensor, keyword_logits: TorchTensor | None
    ) -> list[float] | None: ...


class SentenceTransformerClassifier(nn.Module):
    embedding_dim: int
    hidden_dim: int
    num_classes: int
    dropout: float

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class TransformerIntentModel(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    model: Any


class EmotionallyAdaptiveModel(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class IntentClassifier(nn.Module):
    label_to_idx: Mapping[str, int]
    vocab_size: int
    encoder_type: str

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


def evaluate(
    model: nn.Module,
    dataloader: Iterable[Sequence[Any]],
    criterion: Any,
    device: TorchDevice,
    *,
    return_details: bool = False,
    emotion_config: EmotionTrainingConfig | None = None,
    meta_stacker: MetaIntentStacker | None = None,
) -> tuple[float, float] | tuple[float, float, list[int], list[int], list[list[float]]]: ...


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: Any,
    optimizer: TorchOptimizer,
    device: TorchDevice,
    *,
    scaler: Any = ...,  # GradScaler-like protocol
    epoch: int = ...,
    total_epochs: int = ...,
    log_interval: int = ...,
) -> tuple[float, float]: ...


def build_vocab(
    texts: Sequence[str],
    labels: Sequence[str] | None = ...,
    *,
    config: VocabularyConfig | None = ...,
    min_freq: int = ...,
    extra_texts: Sequence[str] | None = ...,
) -> dict[str, int]: ...


def read_dataset(path: Path) -> tuple[list[str], list[str], list[dict[str, str]]]: ...


class ResponseOutcome:
    message: str
    strategy: str
    basis: str | None


def generate_response(label: str, text: str) -> ResponseOutcome: ...


class ModelPrediction:
    label: str
    confidence: float
    top_predictions: list[tuple[str, float]]


def predict_with_trace(
    model: nn.Module,
    text: str,
    *,
    vocab: Mapping[str, int] | None = ...,
    label_to_idx: Mapping[str, int] | None = ...,
    max_len: int | None = ...,
    device: TorchDevice | None = ...,
    tokenizer: Any | None = ...,
    tokenizer_cache: Any | None = ...,
    embedding_model: Any | None = ...,
    top_k: int | None = ...,
) -> Any: ...


def main(argv: Sequence[str] | None = ...) -> None: ...


def _is_colab_environment() -> bool: ...


def _colab_workspace_root() -> Path: ...
