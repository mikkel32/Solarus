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


class EmotionTrainingConfig:
    enabled: bool


class MetaIntentStacker:
    def compute_adjustment(
        self, base_logits: TorchTensor, keyword_logits: Optional[TorchTensor]
    ) -> Optional[List[float]]: ...


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


def main(argv: Optional[Sequence[str]] = ...) -> None: ...

def _is_colab_environment() -> bool: ...

def _colab_workspace_root() -> Path: ...
