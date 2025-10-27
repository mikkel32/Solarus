from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
        self, base_logits: TorchTensor, keyword_logits: Optional[TorchTensor]
    ) -> Optional[List[float]]: ...


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
    emotion_config: Optional[EmotionTrainingConfig] = None,
    meta_stacker: Optional[MetaIntentStacker] = None,
) -> Tuple[float, float] | Tuple[float, float, List[int], List[int], List[List[float]]]: ...


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
) -> Tuple[float, float]: ...


def build_vocab(
    texts: Sequence[str],
    labels: Sequence[str] | None = ...,
    *,
    config: VocabularyConfig | None = ...,
    min_freq: int = ...,
    extra_texts: Sequence[str] | None = ...,
) -> Dict[str, int]: ...


def read_dataset(path: Path) -> Tuple[List[str], List[str], List[Dict[str, str]]]: ...


class ResponseOutcome:
    message: str
    strategy: str
    basis: Optional[str]


def generate_response(label: str, text: str) -> ResponseOutcome: ...


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


def main(argv: Optional[Sequence[str]] = ...) -> None: ...


def _is_colab_environment() -> bool: ...


def _colab_workspace_root() -> Path: ...
