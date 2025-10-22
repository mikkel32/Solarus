"""Train an intent classification model with advanced optimisation features.

This script builds a text classification model using PyTorch. It reads the CSV
dataset in ``data/intent_dataset.csv``, tokenises the text, and trains an
encoder (BiLSTM, transformer, or sentence-transformer head) before saving the
weights and metadata to the ``models`` directory.

Compared to the initial baseline, the trainer now supports modern optimisation
features: AdamW with configurable weight decay, label smoothing, gradient
clipping, optional mixed-precision training, cosine/One-Cycle schedulers, early
stopping, and a self-training loop that can pseudo-label unlabelled examples to
keep improving over time. A stratified cross-validation harness exercises every
labelled example, and the pseudo-labelling routine now decays its confidence
threshold while weighting samples by their confidence so that the model keeps
learning in a controlled, self-improving fashion.

The module also retains a lightweight response generator so that, once trained,
the model can categorise user input and craft a short natural-language reply
that fits the detected intent.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import contextlib
import unicodedata
import hashlib
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:  # Python 3.11+ exposes datetime.UTC; provide a fallback for older runtimes.
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - compatibility shim for Python < 3.11
    UTC = timezone.utc  # type: ignore[assignment]

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn


def _build_amp_helpers():  # pragma: no cover - helper to keep AMP optional.
    try:  # Prefer the newer torch.amp API when available.
        from torch.amp import GradScaler as _GradScaler  # type: ignore[attr-defined]
        from torch.amp import autocast as _autocast  # type: ignore[attr-defined]

        def create_scaler(enabled: bool):
            return _GradScaler(device="cuda", enabled=enabled)

        def autocast_context(enabled: bool):
            return _autocast(device_type="cuda", enabled=enabled)

        return _GradScaler, create_scaler, autocast_context
    except (ImportError, TypeError):
        try:
            from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore[attr-defined]
            from torch.cuda.amp import autocast as _autocast  # type: ignore[attr-defined]

            def create_scaler(enabled: bool):
                return _GradScaler(enabled=enabled)

            def autocast_context(enabled: bool):
                return _autocast(enabled=enabled)

            return _GradScaler, create_scaler, autocast_context
        except ImportError:
            def create_scaler(enabled: bool):
                return None

            def autocast_context(enabled: bool):
                return contextlib.nullcontext()

            return None, create_scaler, autocast_context


GradScaler, create_grad_scaler, autocast_context = _build_amp_helpers()

TRAINER_VERSION = "orion-trainer-0.4"

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def compute_sha1(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class ModelRegistry:
    def __init__(self, root: Path, model_name: str, *, tolerance: float = 1e-4) -> None:
        self.root = root
        self.model_name = model_name
        self.tolerance = tolerance
        self.best_dir = self.root / self.model_name
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def best_accuracy(self) -> Optional[float]:
        metrics_path = self.best_dir / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            try:
                return float(data.get("validation_accuracy"))
            except (TypeError, ValueError):
                return None
        return None

    def is_improvement(self, val_accuracy: float) -> bool:
        current_best = self.best_accuracy()
        if current_best is None:
            return True
        return val_accuracy > current_best + self.tolerance

    def create_run_directory(self, val_accuracy: float, tag: Optional[str] = None) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        acc_fragment = f"acc{val_accuracy * 100:.2f}".replace(".", "p")
        parts = [timestamp, acc_fragment]
        if tag:
            safe_tag = re.sub(r"[^A-Za-z0-9_-]+", "-", tag.strip()) or "run"
            parts.append(safe_tag.lower())
        run_dir = self.runs_dir / "__".join(parts)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def promote(self, source_dir: Path) -> None:
        if self.best_dir.exists():
            shutil.rmtree(self.best_dir)
        shutil.copytree(source_dir, self.best_dir)


def load_transformer_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer, logging as transformers_logging
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'transformers' package is required for the transformer encoder. "
            "Install it via 'pip install transformers'."
        ) from exc
    transformers_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'sentence-transformers' package is required for the st encoder. "
            "Install it via 'pip install sentence-transformers'."
        ) from exc
    return SentenceTransformer(model_name)


def read_dataset(path: Path) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip()
            label = row["label"].strip()
            if text and label:
                texts.append(text)
                labels.append(label)
    return texts, labels


def read_unlabeled_dataset(path: Path) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "text" not in reader.fieldnames:
            raise ValueError(f"Unlabeled dataset at {path} must contain a 'text' column.")
        for row in reader:
            text = row["text"].strip()
            if text:
                texts.append(text)
    return texts


def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.lower()


def tokenize(text: str) -> List[str]:
    normalised = normalise_text(text)
    return TOKEN_PATTERN.findall(normalised)


def build_vocab(texts: Sequence[str], min_freq: int = 1) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    tokens = tokenize(text)
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens[:max_len]]
    if len(token_ids) < max_len:
        token_ids.extend([vocab[PAD_TOKEN]] * (max_len - len(token_ids)))
    return token_ids


def stratified_split(indices: Sequence[int], labels: Sequence[str], *,
                     test_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    by_label: Dict[str, List[int]] = defaultdict(list)
    for idx, label in zip(indices, labels):
        by_label[label].append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    test_indices: List[int] = []
    for label_indices in by_label.values():
        rng.shuffle(label_indices)
        split = max(1, int(round(len(label_indices) * (1 - test_ratio))))
        train_indices.extend(label_indices[:split])
        test_indices.extend(label_indices[split:])
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def stratified_kfold(indices: Sequence[int], labels: Sequence[str], *,
                     n_splits: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for k-fold cross-validation.")
    by_label: Dict[str, List[int]] = defaultdict(list)
    for idx, label in zip(indices, labels):
        by_label[label].append(idx)

    rng = random.Random(seed)
    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for label_indices in by_label.values():
        rng.shuffle(label_indices)
        for position, example_idx in enumerate(label_indices):
            folds[position % n_splits].append(example_idx)

    splits: List[Tuple[List[int], List[int]]] = []
    for fold_idx in range(n_splits):
        val_indices = sorted(folds[fold_idx])
        train_indices = sorted(
            example_idx
            for other_idx, fold in enumerate(folds)
            if other_idx != fold_idx
            for example_idx in fold
        )
        splits.append((train_indices, val_indices))
    return splits


def compute_round_threshold(base_threshold: float, round_idx: int,
                            decay: float, min_threshold: float) -> float:
    if round_idx <= 1 or decay <= 0:
        return max(min_threshold, base_threshold)
    adjusted = base_threshold * ((1.0 - decay) ** (round_idx - 1))
    return max(min_threshold, adjusted)


def compute_pseudo_weight(base_weight: float, confidence: float, threshold: float,
                          power: float, max_multiplier: float) -> float:
    safe_threshold = max(threshold, 1e-6)
    ratio = max(confidence / safe_threshold, 1.0)
    weight = base_weight * (ratio ** power if power != 0 else 1.0)
    return min(weight, base_weight * max_multiplier)


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    safe_temperature = max(temperature, 1e-4)
    student_log_probs = F.log_softmax(student_logits / safe_temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / safe_temperature, dim=-1)
    kd = F.kl_div(student_log_probs, teacher_probs, reduction="none")
    if kd.dim() == 2:
        kd = kd.sum(dim=1)
    return kd * (safe_temperature ** 2)


def create_ema_model(model: nn.Module, decay: float) -> AveragedModel:
    def ema_average(param_avg: torch.Tensor, param: torch.Tensor, num_averaged: int) -> torch.Tensor:
        if num_averaged == 0:
            return param.detach()
        return param_avg * decay + param.detach() * (1.0 - decay)

    ema_model = AveragedModel(model, avg_fn=ema_average)
    ema_model.module.load_state_dict(model.state_dict())
    return ema_model


def clone_model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Detach and copy a model's parameters to CPU, unwrapping wrappers first."""

    module = model
    seen: Set[int] = set()
    while True:
        module_id = id(module)
        if module_id in seen:
            break
        seen.add(module_id)
        if isinstance(module, AveragedModel):
            module = module.module
            continue
        child = getattr(module, "module", None)
        if isinstance(child, nn.Module) and child is not module:
            module = child
            continue
        break

    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def apply_augmentation_strategy(tokens: List[str], strategy: str, rng: random.Random) -> List[str]:
    if not tokens:
        return tokens
    modified = tokens[:]
    strategy = strategy.lower()
    if strategy == "swap" and len(modified) >= 2:
        idx1, idx2 = rng.sample(range(len(modified)), 2)
        modified[idx1], modified[idx2] = modified[idx2], modified[idx1]
    elif strategy == "delete" and len(modified) >= 2:
        del modified[rng.randrange(len(modified))]
    elif strategy == "duplicate":
        idx = rng.randrange(len(modified))
        modified.insert(idx, modified[idx])
    elif strategy == "rotate" and len(modified) >= 2:
        shift = rng.randrange(1, len(modified))
        modified = modified[shift:] + modified[:shift]
    elif strategy == "shuffle" and len(modified) >= 3:
        middle = modified[1:-1]
        rng.shuffle(middle)
        modified = [modified[0], *middle, modified[-1]]
    elif strategy == "mask":
        idx = rng.randrange(len(modified))
        modified[idx] = "<mask>"
    return modified


def augment_training_corpus(
    texts: Sequence[str],
    labels: Sequence[str],
    weights: Sequence[float],
    *,
    probability: float,
    strategies: Sequence[str],
    max_copies: int,
    max_transforms: int,
    rng: random.Random,
) -> Tuple[List[str], List[str], List[float], int]:
    if probability <= 0 or max_copies <= 0 or not strategies:
        return list(texts), list(labels), list(weights), 0
    augmented_texts = list(texts)
    augmented_labels = list(labels)
    augmented_weights = list(weights)
    augment_count = 0
    for text, label, weight in zip(texts, labels, weights):
        base_tokens = tokenize(text)
        if not base_tokens:
            continue
        for _ in range(max_copies):
            if rng.random() > probability:
                continue
            tokens = base_tokens[:]
            transforms = rng.randint(1, max(1, max_transforms))
            for _ in range(transforms):
                strategy = rng.choice(strategies)
                tokens = apply_augmentation_strategy(tokens, strategy, rng)
            augmented_text = " ".join(tokens).strip()
            if not augmented_text or augmented_text == text:
                continue
            augmented_texts.append(augmented_text)
            augmented_labels.append(label)
            augmented_weights.append(weight)
            augment_count += 1
    return augmented_texts, augmented_labels, augmented_weights, augment_count


def compute_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    *,
    label_to_idx: Dict[str, int],
) -> Dict[str, object]:
    total = len(targets)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    confusion: Dict[str, Dict[str, int]] = {
        idx_to_label[idx]: {idx_to_label[jdx]: 0 for jdx in idx_to_label}
        for idx in idx_to_label
    }
    per_label_counts: Dict[str, Dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0}
        for label in idx_to_label.values()
    }
    for target, pred in zip(targets, predictions):
        target_label = idx_to_label[target]
        pred_label = idx_to_label[pred]
        confusion[target_label][pred_label] += 1
        if target == pred:
            per_label_counts[target_label]["tp"] += 1
        else:
            per_label_counts[pred_label]["fp"] += 1
            per_label_counts[target_label]["fn"] += 1

    per_label_metrics: Dict[str, Dict[str, float]] = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    label_support: Dict[str, int] = {idx_to_label[idx]: 0 for idx in idx_to_label}
    for target in targets:
        label_support[idx_to_label[target]] += 1

    for label, counts in per_label_counts.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        per_label_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": label_support[label],
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_precision += precision * label_support[label]
        weighted_recall += recall * label_support[label]
        weighted_f1 += f1 * label_support[label]

    num_labels = max(1, len(per_label_metrics))
    macro_precision /= num_labels
    macro_recall /= num_labels
    macro_f1 /= num_labels
    support_total = max(1, total)
    weighted_precision /= support_total
    weighted_recall /= support_total
    weighted_f1 /= support_total

    accuracy = sum(int(t == p) for t, p in zip(targets, predictions)) / max(1, total)
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "support": support_total,
        "per_label": per_label_metrics,
        "confusion_matrix": confusion,
    }


@dataclass
class EncodedExample:
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    label: int
    weight: float
    teacher_logits: torch.Tensor


@dataclass
class FoldResult:
    fold_index: int
    val_accuracy: float
    train_accuracy_at_best: float
    metrics: Dict[str, object]
    metadata: Dict[str, object]
    history: List[Dict[str, object]]
    pseudo_rounds: List[Dict[str, object]]
    total_pseudo_added: int
    pseudo_examples: List[Tuple[str, str, float]]
    model_state: Dict[str, torch.Tensor]
    sample_outputs: List[Dict[str, str]]
    run_tag_suffix: Optional[str]


@dataclass
class DistillationConfig:
    alpha: float
    temperature: float


class IntentDataset(Dataset[EncodedExample]):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        vocab: Dict[str, int],
        label_to_idx: Dict[str, int],
        max_len: int,
        sample_weights: Optional[Sequence[float]] = None,
        tokenizer=None,
        tokenizer_cache: Optional[Callable[[str], Tuple[Sequence[int], Sequence[int]]]] = None,
        embedding_model: Optional[Callable[[str], Sequence[float]]] = None,
        teacher_logits: Optional[Sequence[Optional[Sequence[float]]]] = None,
    ) -> None:
        self.examples: List[EncodedExample] = []
        if sample_weights is None:
            sample_weights = [1.0] * len(texts)
        if len(sample_weights) != len(texts):
            raise ValueError("Sample weights must match the number of texts.")
        if teacher_logits is None:
            teacher_iter: Sequence[Optional[Sequence[float]]] = [None] * len(texts)
        else:
            if len(teacher_logits) != len(texts):
                raise ValueError("Teacher logits must align with the provided texts.")
            teacher_iter = teacher_logits
        for text, label, weight, teacher_row in zip(texts, labels, sample_weights, teacher_iter):
            if embedding_model is not None:
                vector = embedding_model(text)
                token_tensor = torch.tensor(vector, dtype=torch.float32)
                mask_tensor = torch.ones_like(token_tensor)
            elif tokenizer_cache is not None:
                cached_ids, cached_mask = tokenizer_cache(text)
                token_tensor = torch.tensor(cached_ids, dtype=torch.long)
                mask_tensor = torch.tensor(cached_mask, dtype=torch.long)
            elif tokenizer is not None:
                encoded = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_attention_mask=True,
                )
                token_tensor = torch.tensor(encoded["input_ids"], dtype=torch.long)
                mask_tensor = torch.tensor(encoded["attention_mask"], dtype=torch.long)
            else:
                encoded = encode_text(text, vocab, max_len)
                token_tensor = torch.tensor(encoded, dtype=torch.long)
                pad_idx = vocab.get(PAD_TOKEN, 0)
                mask_tensor = (token_tensor != pad_idx).long()
            label_id = label_to_idx[label]
            if teacher_row is not None:
                teacher_tensor = torch.tensor(list(teacher_row), dtype=torch.float32)
            else:
                teacher_tensor = torch.empty(0, dtype=torch.float32)
            self.examples.append(
                EncodedExample(
                    tokens=token_tensor,
                    attention_mask=mask_tensor,
                    label=label_id,
                    weight=float(weight),
                    teacher_logits=teacher_tensor,
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        return (
            example.tokens,
            torch.tensor(example.label, dtype=torch.long),
            torch.tensor(example.weight, dtype=torch.float32),
            example.attention_mask,
            example.teacher_logits,
        )


class IntentClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_classes: int, dropout: float = 0.3, num_layers: int = 1,
                 attention_heads: int = 4, ffn_dim: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim * 2),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 6),
            nn.Linear(hidden_dim * 6, hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, num_classes),
        )

    def forward(self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        embedded = self.embedding_dropout(self.embedding(inputs))
        outputs, (hidden, _) = self.lstm(embedded)
        attn_output, _ = self.attention(outputs, outputs, outputs)
        context = self.layer_norm(attn_output + outputs)
        ffn_output = self.layer_norm(self.ffn(context) + context)

        if self.lstm.bidirectional:
            forward_last = hidden[-2]
            backward_last = hidden[-1]
            last_hidden = torch.cat([forward_last, backward_last], dim=1)
        else:
            last_hidden = hidden[-1]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            mask_sum = mask.sum(dim=1).clamp(min=1.0)
            pooled_mean = (ffn_output * mask).sum(dim=1) / mask_sum
            masked_ffn = ffn_output.masked_fill(mask == 0, float("-inf"))
            pooled_max = masked_ffn.max(dim=1).values
            pooled_max[torch.isinf(pooled_max)] = 0.0
        else:
            pooled_mean = ffn_output.mean(dim=1)
            pooled_max, _ = ffn_output.max(dim=1)
        representation = torch.cat([last_hidden, pooled_mean, pooled_max], dim=1)
        logits = self.classifier(representation)
        return logits


class TransformerIntentModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'transformers' package is required for the transformer encoder. "
                "Install it via 'pip install transformers'."
            ) from exc
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
        return outputs.logits


class SentenceTransformerClassifier(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.classifier(inputs)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scheduler_step_per_batch: bool = False,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: Optional[float] = None,
    distillation_config: Optional[DistillationConfig] = None,
) -> Tuple[float, float, Dict[str, int]]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    amp_enabled = scaler is not None and getattr(scaler, "is_enabled", lambda: False)()

    grad_accumulation_steps = max(1, getattr(optimizer, "grad_accumulation_steps", 1))
    ema_model: Optional[AveragedModel] = getattr(optimizer, "ema_model", None)
    ema_active: bool = bool(getattr(optimizer, "ema_active", False))
    swa_model: Optional[AveragedModel] = getattr(optimizer, "swa_model", None)
    swa_active: bool = bool(getattr(optimizer, "swa_active", False))

    optimizer_steps = 0
    ema_updates = 0
    swa_updates = 0

    optimizer.zero_grad(set_to_none=True)
    num_batches = len(dataloader)
    if num_batches == 0:
        return 0.0, 0.0, {"optimizer_steps": 0, "ema_updates": 0, "swa_updates": 0}

    for batch_idx, batch in enumerate(dataloader, start=1):
        if len(batch) == 5:
            inputs, targets, weights, attention_mask, teacher_logits = batch
        else:
            inputs, targets, weights, attention_mask = batch  # type: ignore[misc]
            teacher_logits = None
        inputs = inputs.to(device)
        targets = targets.to(device)
        weights = weights.to(device)
        attention_mask = attention_mask.to(device)
        if teacher_logits is not None:
            teacher_logits = teacher_logits.to(device)

        context = autocast_context(amp_enabled)

        with context:
            logits = model(inputs, attention_mask=attention_mask)
            hard_loss = criterion(logits, targets)
            if hard_loss.dim() == 0:
                hard_loss = hard_loss.unsqueeze(0)
            if (
                distillation_config is not None
                and teacher_logits is not None
                and teacher_logits.numel() > 0
            ):
                kd_loss = compute_distillation_loss(
                    logits,
                    teacher_logits,
                    temperature=distillation_config.temperature,
                )
                loss_values = (
                    hard_loss * (1.0 - distillation_config.alpha)
                    + kd_loss * distillation_config.alpha
                )
            else:
                loss_values = hard_loss
            weight_denominator = weights.sum()
            if float(weight_denominator.item()) == 0.0:
                weight_denominator = torch.tensor(float(loss_values.numel()), device=device)
            weighted_loss = (loss_values * weights).sum() / weight_denominator

        raw_batch_loss = hard_loss.detach().mean().item()
        total_loss += raw_batch_loss * targets.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

        loss_for_backprop = weighted_loss / grad_accumulation_steps

        if amp_enabled and scaler is not None:
            scaler.scale(loss_for_backprop).backward()
        else:
            loss_for_backprop.backward()

        should_step = (batch_idx % grad_accumulation_steps == 0) or (batch_idx == num_batches)

        if should_step:
            if amp_enabled and scaler is not None:
                if max_grad_norm and max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm and max_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and scheduler_step_per_batch:
                scheduler.step()
            if ema_model is not None and ema_active:
                ema_model.update_parameters(model)
                ema_updates += 1
            if swa_model is not None and swa_active:
                swa_model.update_parameters(model)
                swa_updates += 1

    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
        {"optimizer_steps": optimizer_steps, "ema_updates": ema_updates, "swa_updates": swa_updates},
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    return_details: bool = False,
) -> Tuple[float, float] | Tuple[float, float, List[int], List[int], List[List[float]]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    detailed_targets: List[int] = []
    detailed_predictions: List[int] = []
    detailed_probabilities: List[List[float]] = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 5:
                inputs, targets, _weights, attention_mask, _teacher_logits = batch
            else:
                inputs, targets, _weights, attention_mask = batch  # type: ignore[misc]
            inputs = inputs.to(device)
            targets = targets.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(inputs, attention_mask=attention_mask)
            loss_values = criterion(logits, targets)
            if loss_values.dim() == 0:
                loss_values = loss_values.unsqueeze(0)

            batch_loss = loss_values.mean().item()
            total_loss += batch_loss * targets.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            detailed_targets.extend(targets.cpu().tolist())
            detailed_predictions.extend(predictions.cpu().tolist())
            detailed_probabilities.extend(torch.softmax(logits, dim=-1).cpu().tolist())
    if return_details:
        return (
            total_loss / max(total, 1),
            correct / max(total, 1),
            detailed_targets,
            detailed_predictions,
            detailed_probabilities,
        )
    return total_loss / max(total, 1), correct / max(total, 1)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    steps_per_epoch: int,
    max_lr: float,
) -> Tuple[Optional[torch.optim.lr_scheduler._LRScheduler], bool]:
    if scheduler_type == "onecycle" and epochs > 0 and steps_per_epoch > 0:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
        return scheduler, True
    if scheduler_type == "cosine" and epochs > 0:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        return scheduler, False
    return None, False


def pseudo_label_unlabeled(
    model: nn.Module,
    texts: Sequence[str],
    *,
    vocab: Dict[str, int],
    label_to_idx: Dict[str, int],
    max_len: int,
    device: torch.device,
    threshold: float,
    tokenizer=None,
    tokenizer_cache: Optional[Callable[[str], Tuple[Sequence[int], Sequence[int]]]] = None,
    embedding_model: Optional[Callable[[str], Sequence[float]]] = None,
) -> Tuple[List[Tuple[str, str, float]], List[str]]:
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    confident: List[Tuple[str, str, float]] = []
    remaining: List[str] = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            if embedding_model is not None:
                vector = embedding_model(text)
                ids = torch.tensor(vector, dtype=torch.float32, device=device).unsqueeze(0)
                mask = torch.ones_like(ids)
            elif tokenizer_cache is not None:
                cached_ids, cached_mask = tokenizer_cache(text)
                ids = torch.tensor(cached_ids, dtype=torch.long, device=device).unsqueeze(0)
                mask = torch.tensor(cached_mask, dtype=torch.long, device=device).unsqueeze(0)
            elif tokenizer is not None:
                encoded = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_attention_mask=True,
                )
                ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
                mask = torch.tensor(encoded["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
            else:
                raw_ids = encode_text(text, vocab, max_len)
                ids = torch.tensor(raw_ids, dtype=torch.long, device=device).unsqueeze(0)
                pad_idx = vocab.get(PAD_TOKEN, 0)
                mask_values = [1 if token != pad_idx else 0 for token in raw_ids]
                mask = torch.tensor(mask_values, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(ids, attention_mask=mask)
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted = probs.max(dim=-1)
            score = confidence.item()
            if score >= threshold:
                label = idx_to_label[predicted.item()]
                confident.append((text, label, score))
            else:
                remaining.append(text)
    return confident, remaining


def predict_label(
    model: nn.Module,
    text: str,
    *,
    vocab: Dict[str, int],
    label_to_idx: Dict[str, int],
    max_len: int,
    device: torch.device,
    tokenizer=None,
    tokenizer_cache: Optional[Callable[[str], Tuple[Sequence[int], Sequence[int]]]] = None,
    embedding_model: Optional[Callable[[str], Sequence[float]]] = None,
) -> str:
    if embedding_model is not None:
        vector = embedding_model(text)
        ids = torch.tensor(vector, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.ones_like(ids)
    elif tokenizer_cache is not None:
        cached_ids, cached_mask = tokenizer_cache(text)
        ids = torch.tensor(cached_ids, dtype=torch.long, device=device).unsqueeze(0)
        mask = torch.tensor(cached_mask, dtype=torch.long, device=device).unsqueeze(0)
    elif tokenizer is not None:
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_attention_mask=True,
        )
        ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        mask = torch.tensor(encoded["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
    else:
        raw_ids = encode_text(text, vocab, max_len)
        ids = torch.tensor(raw_ids, dtype=torch.long, device=device).unsqueeze(0)
        pad_idx = vocab.get(PAD_TOKEN, 0)
        mask_values = [1 if token != pad_idx else 0 for token in raw_ids]
        mask = torch.tensor(mask_values, dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(ids, attention_mask=mask)
        predicted = logits.argmax(dim=1).item()
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return idx_to_label[predicted]


def answer_question(text: str) -> str:
    lowered = normalise_text(text)
    if "meeting" in lowered and "time" in lowered:
        return "The meeting is scheduled to begin at 10:00 AM tomorrow."
    if "author" in lowered:
        return "You can usually find the author's name on the cover or inside the first few pages."
    if "pharmacy" in lowered or "nearest" in lowered:
        return "There's a pharmacy a couple of blocks away near the main street."
    if "reset" in lowered and "router" in lowered:
        return "Press and hold the router's reset button for about ten seconds until the lights flash."
    if "seat" in lowered and "taken" in lowered:
        return "It looks free—feel welcome to take the seat."
    if "cats" in lowered and "dark" in lowered:
        return "Cats can't see in total darkness, but they can see very well in low light conditions."
    if "clarify" in lowered and "request" in lowered:
        return "Sure—let me know which part you'd like more details on."
    if "reschedule" in lowered and "meeting" in lowered:
        return "Yes, let's find a new time that suits everyone."
    return "That's a great question. I'll investigate and follow up with more details shortly."


DEFAULT_RESPONSES: Dict[str, str] = {
    "greeting": "Hello! It's great to hear from you.",
    "farewell": "Goodbye for now—talk to you soon!",
    "thank_you": "You're welcome! I'm happy to help.",
    "apology": "No worries—thanks for letting me know.",
    "compliment": "Thank you! That means a lot.",
    "criticism": "I appreciate the honest feedback and will work on it.",
    "suggestion": "That's a thoughtful suggestion—I’ll consider it carefully.",
    "request": "I'll take care of that request for you.",
    "instruction": "Got it—I'll follow those instructions.",
    "positive_statement": "That sounds wonderful!",
    "positive_experience": "What an exciting experience!",
    "technical_statement": "Thanks for the technical update—everything looks good.",
    "story_snippet": "What a vivid scene—you paint quite a picture!",
    "poem_line": "That's a lovely poetic line.",
    "news_headline": "Thanks for the update—it's good to stay informed.",
    "joke": "Haha, that's a good one!",
    "humor": "Thanks for the laugh!",
    "fact": "Interesting fact—I'll remember that.",
    "weather_report": "Thanks for the weather update—I’ll plan accordingly.",
    "weather_statement": "Appreciate the weather heads-up!",
    "technical_instruction": "I'll keep that technical guidance in mind.",
    "error_message": "I'll look into resolving that error right away.",
    "motivation": "That's motivating—thanks for the encouragement!",
    "sarcasm": "I sense a bit of sarcasm there!",
    "riddle": "That's a clever riddle—I’ll think on it.",
    "saying": "That's a wise saying.",
    "announcement": "Thanks for the announcement—I'll spread the word.",
    "definition": "Thanks for clarifying that definition.",
    "recommendation": "I appreciate the recommendation.",
    "reminder": "Thanks for the reminder—I won't forget.",
    "observation": "That's an interesting observation.",
    "quote": "That's an inspiring quote.",
    "pun": "I love a good pun!",
    "statement": "Thanks for the update.",
    "advice": "That's helpful advice—much appreciated.",
}


def generate_response(label: str, text: str) -> str:
    if label == "question":
        return answer_question(text)
    if label == "request":
        return f"I'll handle this: {text.strip()}"
    if label == "reminder":
        return f"Reminder received: {text.strip()}"
    if label == "observation":
        return f"Thanks for pointing that out: {text.strip()}"
    if label == "recommendation":
        return f"That recommendation sounds valuable—I'll keep it in mind."
    if label == "statement":
        return f"Understood: {text.strip()}"
    if label == "announcement":
        return f"Announcement noted: {text.strip()}"
    if label == "definition":
        return f"That's a clear definition—thanks for sharing."
    if label == "advice":
        return f"I'll follow that advice: {text.strip()}"
    if label == "riddle":
        return "I'll give it some thought—riddles keep the mind sharp!"
    if label == "sarcasm":
        return "I hear the sarcasm—let's make the best of it anyway."
    return DEFAULT_RESPONSES.get(label, f"I recognized this as a '{label}' message.")


def write_accuracy_readme(
    target_dir: Path,
    model_name: str,
    metrics: Dict[str, object],
    *,
    tolerance: float,
) -> None:
    accuracy = float(metrics.get("validation_accuracy", 0.0)) * 100
    train_accuracy = float(metrics.get("train_accuracy_at_best", 0.0)) * 100
    dataset_examples = int(metrics.get("dataset_examples", 0))
    pseudo_examples = int(metrics.get("pseudo_examples_added", 0))
    best_epoch = metrics.get("best_epoch", "-")
    best_stage = metrics.get("best_stage", "-")
    timestamp = metrics.get("timestamp_utc", "-")
    checksum = metrics.get("dataset_checksum", "-")
    num_labels = metrics.get("num_labels", "-")
    promoted = bool(metrics.get("promoted_to_orion", False))
    encoder_type = str(metrics.get("encoder_type", "unknown"))
    transformer_model = metrics.get("transformer_model")
    sentence_model_name = metrics.get("sentence_transformer_model")
    learning_rate = float(metrics.get("effective_learning_rate", 0.0))
    header_name = model_name.replace("_", " ").title()
    encoder_line = f"- **Encoder type:** {encoder_type}"
    if transformer_model:
        encoder_line += f" ({transformer_model})"
    elif sentence_model_name:
        encoder_line += f" ({sentence_model_name})"
    lines = [
        f"# {header_name} snapshot",
        "",
        f"- **Validation accuracy:** {accuracy:.2f}%",
        f"- **Training accuracy at best epoch:** {train_accuracy:.2f}%",
        f"- **Best epoch/stage:** {best_epoch} ({best_stage})",
        f"- **Labelled dataset:** {dataset_examples} examples across {num_labels} intents",
        f"- **Pseudo-labelled additions:** {pseudo_examples}",
        encoder_line,
        f"- **Effective learning rate:** {learning_rate:.2e}",
        f"- **Dataset checksum:** `{checksum}`",
        f"- **Trainer version:** {metrics.get('trainer_version', '-')}",
        f"- **Run timestamp (UTC):** {timestamp}",
        f"- **Promoted to {model_name}:** {'yes' if promoted else 'no'}",
        "",
        "## Promotion rules",
        f"- Promote a run to `{model_name}` only when its validation accuracy exceeds the previous best by more than {tolerance * 100:.4f} percentage points.",
        "- Keep every training run under `models/runs/` for auditing and reproducibility.",
        "- Update the metadata and metrics files alongside the weights when a promotion occurs.",
    ]
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def save_run_artifacts(
    model: nn.Module,
    metadata: Dict[str, object],
    metrics: Dict[str, object],
    run_dir: Path,
    *,
    model_name: str,
    tolerance: float,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "model.pt"
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.json"
    torch.save(model.state_dict(), weights_path)
    write_json(metadata_path, metadata)
    write_json(metrics_path, metrics)
    write_accuracy_readme(run_dir, model_name, metrics, tolerance=tolerance)
    print(f"Saved run artifacts to {run_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an intent classification model with advanced optimisation, cross-validation, and self-learning capabilities."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/intent_dataset.csv"),
                        help="Path to the labelled CSV dataset containing 'text' and 'label' columns.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"),
                        help="Directory where model artefacts will be recorded.")
    parser.add_argument("--model-name", type=str, default="orion_v0.4",
                        help="Name of the high-watermark model directory inside --model-dir.")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Optional label appended to run directories for easier identification.")
    parser.add_argument("--promotion-tolerance", type=float, default=1e-4,
                        help="Minimum absolute validation-accuracy improvement required to promote Orion.")
    parser.add_argument("--encoder-type", choices=["bilstm", "transformer", "st"], default="transformer",
                        help="Select between the BiLSTM encoder and a pretrained transformer backbone.")
    parser.add_argument("--transformer-model", type=str, default="prajjwal1/bert-tiny",
                        help="Hugging Face model checkpoint to fine-tune when --encoder-type=transformer.")
    parser.add_argument("--transformer-learning-rate", type=float, default=1e-4,
                        help="Learning rate used when fine-tuning the transformer encoder.")
    parser.add_argument("--sentence-transformer-model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformer checkpoint for frozen-embedding classification when --encoder-type=st.")
    parser.add_argument("--st-hidden-dim", type=int, default=512,
                        help="Hidden dimension of the feed-forward head for sentence-transformer embeddings.")
    parser.add_argument("--st-dropout", type=float, default=0.2,
                        help="Dropout applied in the sentence-transformer feed-forward head.")
    parser.add_argument("--embedding-dim", type=int, default=160,
                        help="Size of the token embeddings (BiLSTM encoder only).")
    parser.add_argument("--hidden-dim", type=int, default=192,
                        help="Hidden dimension of the BiLSTM encoder.")
    parser.add_argument("--ffn-dim", type=int, default=384,
                        help="Width of the feed-forward layer after attention (BiLSTM encoder only).")
    parser.add_argument("--encoder-layers", type=int, default=2,
                        help="Number of stacked BiLSTM layers.")
    parser.add_argument("--attention-heads", type=int, default=4,
                        help="Number of attention heads for the self-attention block (BiLSTM encoder only).")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate applied throughout the network (BiLSTM encoder only).")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of supervised training epochs.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Peak learning rate for the optimiser/scheduler (BiLSTM encoder).")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay used by AdamW.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Clip gradients to this norm (set <=0 to disable).")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Amount of label smoothing to apply during training.")
    parser.add_argument("--scheduler", choices=["onecycle", "cosine", "none"], default="onecycle",
                        help="Learning-rate scheduler strategy.")
    parser.add_argument("--grad-accumulation-steps", type=int, default=1,
                        help="Number of gradient accumulation steps per optimiser update.")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="Exponential moving-average decay applied to model weights (0 disables).")
    parser.add_argument("--ema-start-epoch", type=int, default=1,
                        help="Global epoch after which EMA updates begin to accumulate.")
    parser.add_argument("--ema-use-for-eval", action="store_true",
                        help="Use EMA weights for validation, pseudo-labelling, and promotion once the EMA is active.")
    parser.add_argument("--swa-start-epoch", type=int, default=0,
                        help="Global epoch at which to begin stochastic weight averaging (0 disables).")
    parser.add_argument("--swa-lr", type=float, default=0.0,
                        help="Learning rate used once SWA is active (0 reuses the optimiser's learning rate).")
    parser.add_argument("--swa-anneal-epochs", type=int, default=5,
                        help="Number of epochs used to anneal into the SWA learning rate.")
    parser.add_argument("--augment-probability", type=float, default=0.0,
                        help="Probability of generating an augmented variant for each training example per epoch.")
    parser.add_argument("--augment-max-copies", type=int, default=1,
                        help="Maximum augmented copies produced per example in each epoch.")
    parser.add_argument("--augment-max-transforms", type=int, default=2,
                        help="Maximum number of token-level transforms applied to any augmented example.")
    parser.add_argument("--augment-strategies", type=str, default="swap,delete,duplicate",
                        help="Comma-separated list of augmentation strategies (swap, delete, duplicate, rotate, shuffle, mask).")
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Stop training after this many epochs without validation improvement (0 disables).")
    parser.add_argument("--min-freq", type=int, default=1,
                        help="Minimum token frequency required to enter the vocabulary.")
    parser.add_argument("--max-seq-len", type=int, default=128,
                        help="Maximum number of tokens retained per example (after tokenisation).")
    parser.add_argument("--self-train-rounds", type=int, default=2,
                        help="Number of self-training refinement rounds.")
    parser.add_argument("--self-train-epochs", type=int, default=2,
                        help="Additional epochs to run after adding pseudo-labelled data in each round.")
    parser.add_argument("--self-train-threshold", type=float, default=0.92,
                        help="Confidence threshold for accepting pseudo-labels.")
    parser.add_argument("--self-train-weight", type=float, default=0.5,
                        help="Loss weight applied to pseudo-labelled examples (relative to 1.0 for gold labels).")
    parser.add_argument("--unlabeled-dataset", type=Path,
                        help="Optional CSV containing an unlabeled 'text' column for self-training.")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable mixed-precision training when CUDA/AMP are available.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Fraction of the dataset to reserve for validation.")
    parser.add_argument("--folds", type=int, default=1,
                        help="Number of stratified folds used for cross-validation (set to 1 for a single split).")
    parser.add_argument("--self-train-threshold-decay", type=float, default=0.05,
                        help="Multiplicative decay applied to the pseudo-labelling confidence threshold after each round (set to 0 to disable).")
    parser.add_argument("--self-train-min-threshold", type=float, default=0.75,
                        help="Lower bound applied to the pseudo-labelling confidence threshold after decay.")
    parser.add_argument("--self-train-confidence-power", type=float, default=1.0,
                        help="Exponent applied to the confidence/threshold ratio when deriving pseudo-label weights.")
    parser.add_argument("--self-train-max-weight-multiplier", type=float, default=3.0,
                        help="Maximum multiple of --self-train-weight permitted for confidence-based weighting.")
    parser.add_argument("--final-train-epochs", type=int, default=0,
                        help="Extra epochs to fine-tune the promoted model on the full labelled dataset after cross-validation.")
    parser.add_argument("--final-train-learning-rate", type=float, default=0.0,
                        help="Override learning rate for the final full-data training stage (0 keeps the previous rate).")
    parser.add_argument("--final-train-weight-decay", type=float, default=-1.0,
                        help="Override weight decay for the final training stage (-1 keeps the previous value).")
    parser.add_argument("--final-train-batch-size", type=int, default=0,
                        help="Override batch size for the final training stage (0 reuses --batch-size).")
    parser.add_argument("--final-train-scheduler", choices=["inherit", "onecycle", "cosine", "none"], default="inherit",
                        help="Scheduler strategy applied during the final full-data training stage.")
    parser.add_argument("--final-use-pseudo", dest="final_use_pseudo", action="store_true",
                        help="Include pseudo-labelled examples gathered across folds in the final training stage.")
    parser.add_argument("--no-final-pseudo", dest="final_use_pseudo", action="store_false",
                        help="Skip pseudo-labelled examples when running the final training stage.")
    parser.set_defaults(final_use_pseudo=True)
    parser.add_argument("--distill-epochs", type=int, default=0,
                        help="Number of knowledge-distillation epochs to run before the final consolidation stage.")
    parser.add_argument("--distill-alpha", type=float, default=0.6,
                        help="Blend factor for teacher probabilities (1.0 uses only teacher targets, 0.0 ignores them).")
    parser.add_argument("--distill-temperature", type=float, default=2.5,
                        help="Softmax temperature applied to teacher/student logits during distillation.")
    parser.add_argument("--distill-min-confidence", type=float, default=0.55,
                        help="Discard teacher predictions below this confidence during distillation.")
    parser.add_argument("--distill-confidence-power", type=float, default=1.25,
                        help="Exponent applied to teacher confidence when scaling distillation sample weights.")
    parser.add_argument("--distill-max-weight-multiplier", type=float, default=4.0,
                        help="Cap on the confidence-derived weight multiplier applied to teacher-guided samples.")
    parser.add_argument("--distill-max-teachers", type=int, default=3,
                        help="Maximum number of fold models ensembled as teachers (0 keeps every fold).")
    parser.add_argument("--distill-keep-during-final", action="store_true",
                        help="Blend distillation loss into the final training stage alongside hard labels.")
    parser.add_argument("--no-distill-keep", dest="distill_keep_during_final", action="store_false",
                        help="Disable distillation loss during the final training stage even if teachers are prepared.")
    parser.set_defaults(distill_keep_during_final=False)
    args = parser.parse_args()

    if not 0.0 < args.self_train_threshold <= 1.0:
        parser.error("--self-train-threshold must be in the interval (0, 1].")
    if args.self_train_weight <= 0:
        parser.error("--self-train-weight must be positive.")
    if args.max_seq_len <= 0:
        parser.error("--max-seq-len must be positive.")
    if args.transformer_learning_rate <= 0:
        parser.error("--transformer-learning-rate must be positive.")
    if args.folds < 1:
        parser.error("--folds must be at least 1.")
    if not 0 <= args.self_train_threshold_decay < 1:
        parser.error("--self-train-threshold-decay must lie in [0, 1).")
    if not 0 < args.self_train_min_threshold <= args.self_train_threshold:
        parser.error("--self-train-min-threshold must be in (0, --self-train-threshold].")
    if args.self_train_confidence_power < 0:
        parser.error("--self-train-confidence-power must be non-negative.")
    if args.self_train_max_weight_multiplier < 1:
        parser.error("--self-train-max-weight-multiplier must be at least 1.")
    if args.grad_accumulation_steps < 1:
        parser.error("--grad-accumulation-steps must be at least 1.")
    if not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must lie in [0, 1).")
    if args.ema_start_epoch < 1:
        parser.error("--ema-start-epoch must be at least 1.")
    if args.swa_start_epoch < 0:
        parser.error("--swa-start-epoch must be non-negative.")
    if args.swa_lr < 0:
        parser.error("--swa-lr must be non-negative.")
    if args.swa_anneal_epochs < 1:
        parser.error("--swa-anneal-epochs must be at least 1.")
    if not 0 <= args.augment_probability <= 1:
        parser.error("--augment-probability must lie in [0, 1].")
    if args.distill_epochs < 0:
        parser.error("--distill-epochs must be non-negative.")
    if not 0 <= args.distill_alpha <= 1:
        parser.error("--distill-alpha must lie in [0, 1].")
    if args.distill_temperature <= 0:
        parser.error("--distill-temperature must be positive.")
    if not 0 <= args.distill_min_confidence <= 1:
        parser.error("--distill-min-confidence must lie in [0, 1].")
    if args.distill_confidence_power < 0:
        parser.error("--distill-confidence-power must be non-negative.")
    if args.distill_max_weight_multiplier < 1:
        parser.error("--distill-max-weight-multiplier must be at least 1.")
    if args.distill_max_teachers < 0:
        parser.error("--distill-max-teachers must be non-negative.")
    if args.augment_max_copies < 0:
        parser.error("--augment-max-copies must be non-negative.")
    if args.augment_max_transforms < 1:
        parser.error("--augment-max-transforms must be at least 1.")
    if args.final_train_epochs < 0:
        parser.error("--final-train-epochs must be non-negative.")
    if args.final_train_learning_rate < 0:
        parser.error("--final-train-learning-rate must be non-negative.")
    if args.final_train_weight_decay < -1:
        parser.error("--final-train-weight-decay must be greater than or equal to -1.")
    if args.final_train_batch_size < 0:
        parser.error("--final-train-batch-size must be non-negative.")

    augment_strategies = [strategy.strip() for strategy in args.augment_strategies.split(",") if strategy.strip()]
    if args.augment_probability > 0 and not augment_strategies:
        parser.error("--augment-probability requires at least one augmentation strategy.")

    set_seed(args.seed)

    if not args.dataset.exists():
        raise FileNotFoundError(f"Labelled dataset not found: {args.dataset}")

    dataset_checksum = compute_sha1(args.dataset)
    texts, labels = read_dataset(args.dataset)
    if not texts:
        raise RuntimeError(f"Dataset at {args.dataset} is empty.")

    folds_requested = max(1, args.folds)
    label_counts = Counter(labels)
    min_label_count = min(label_counts.values())
    folds = min(folds_requested, len(texts))
    if folds > min_label_count:
        print(
            f"Requested {folds} folds but the rarest label only has {min_label_count} examples; "
            f"reducing folds to {min_label_count}."
        )
        folds = max(1, min_label_count)
    if folds != folds_requested:
        print(f"Using {folds} folds for cross-validation instead of the requested {folds_requested}.")

    tokenizer_obj = None
    tokenizer_cache_fn: Optional[Callable[[str], Tuple[Tuple[int, ...], Tuple[int, ...]]]] = None
    embedding_fn: Optional[Callable[[str], Sequence[float]]] = None
    sentence_model = None
    sentence_embedding_dim: Optional[int] = None
    embedding_cache_info: Optional[Callable[[], object]] = None
    if args.encoder_type == "transformer":
        tokenizer_obj = load_transformer_tokenizer(args.transformer_model)
        max_seq_len = args.max_seq_len

        tokenizer_cache_store: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
        chunk_size = max(32, args.batch_size * 4 if args.batch_size > 0 else 128)

        def _encode_batch(samples: Sequence[str]) -> None:
            if not samples:
                return
            encoded = tokenizer_obj(
                list(samples),
                padding="max_length",
                truncation=True,
                max_length=max_seq_len,
                return_attention_mask=True,
            )
            input_ids = encoded["input_ids"]
            attention_masks = encoded["attention_mask"]
            for idx, sample in enumerate(samples):
                if sample in tokenizer_cache_store:
                    continue
                tokenizer_cache_store[sample] = (
                    tuple(int(x) for x in input_ids[idx]),
                    tuple(int(x) for x in attention_masks[idx]),
                )

        def populate_tokenizer_cache(samples: Sequence[str], description: str) -> None:
            unique_samples = [
                sample for sample in dict.fromkeys(samples)
                if sample and sample not in tokenizer_cache_store
            ]
            if not unique_samples:
                return
            print(
                f"Tokenising {len(unique_samples)} texts for {description} "
                f"(batch size {chunk_size})."
            )
            for start in range(0, len(unique_samples), chunk_size):
                batch = unique_samples[start:start + chunk_size]
                _encode_batch(batch)

        def tokenizer_cache_fn(sample: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
            cached = tokenizer_cache_store.get(sample)
            if cached is None:
                _encode_batch([sample])
                cached = tokenizer_cache_store[sample]
            return cached
    elif args.encoder_type == "st":
        sentence_model = load_sentence_transformer(args.sentence_transformer_model)
        sentence_embedding_dim = int(sentence_model.get_sentence_embedding_dimension())

        @lru_cache(maxsize=8192)
        def embed_text_cached(sample: str) -> Tuple[float, ...]:
            vector = sentence_model.encode([sample], show_progress_bar=False)
            return tuple(float(x) for x in vector[0])

        def embed_text(sample: str) -> Sequence[float]:
            return list(embed_text_cached(sample))

        embedding_fn = embed_text
        max_seq_len = 1
        embedding_cache_info = embed_text_cached.cache_info
    else:
        max_tokens = max(len(tokenize(text)) for text in texts)
        max_seq_len = max(8, min(args.max_seq_len, max_tokens))

    vocab = build_vocab(texts, min_freq=args.min_freq)
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(label_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(target_device: Optional[torch.device] = None) -> nn.Module:
        destination = target_device or device
        if args.encoder_type == "transformer":
            model_obj = TransformerIntentModel(args.transformer_model, num_classes)
            if tokenizer_obj is not None and hasattr(model_obj, "model") and hasattr(model_obj.model, "resize_token_embeddings"):
                model_obj.model.resize_token_embeddings(len(tokenizer_obj))
            return model_obj.to(destination)
        if args.encoder_type == "st":
            if sentence_embedding_dim is None:
                raise RuntimeError("Sentence-transformer embedding dimension could not be determined.")
            model_obj = SentenceTransformerClassifier(
                embedding_dim=sentence_embedding_dim,
                hidden_dim=args.st_hidden_dim,
                num_classes=num_classes,
                dropout=args.st_dropout,
            )
            return model_obj.to(destination)
        model_obj = IntentClassifier(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            num_layers=args.encoder_layers,
            attention_heads=args.attention_heads,
            ffn_dim=args.ffn_dim,
        )
        return model_obj.to(destination)

    unlabeled_master: List[str] = []
    unlabeled_checksum: Optional[str] = None
    if args.unlabeled_dataset:
        if not args.unlabeled_dataset.exists():
            raise FileNotFoundError(f"Unlabeled dataset not found: {args.unlabeled_dataset}")
        unlabeled_master = read_unlabeled_dataset(args.unlabeled_dataset)
        unlabeled_checksum = compute_sha1(args.unlabeled_dataset)
        print(f"Loaded {len(unlabeled_master)} unlabeled examples for self-training.")
    elif args.self_train_rounds > 0:
        print("No unlabeled dataset supplied; skipping self-training despite configured rounds.")

    if args.encoder_type == "transformer":
        populate_tokenizer_cache(texts, "labelled dataset")
        if unlabeled_master:
            populate_tokenizer_cache(unlabeled_master, "unlabelled dataset")

    indices = list(range(len(texts)))
    if folds > 1:
        fold_pairs = stratified_kfold(indices, labels, n_splits=folds, seed=args.seed)
        print(f"Running stratified {folds}-fold cross-validation.")
    else:
        train_indices, val_indices = stratified_split(indices, labels,
                                                      test_ratio=args.test_ratio,
                                                      seed=args.seed)
        fold_pairs = [(train_indices, val_indices)]

    total_folds = len(fold_pairs)
    sample_inputs = [
        "Hello there!",
        "Please reboot the system after tonight's update.",
        "What time does the meeting start tomorrow?",
        "Remember to submit your report by Friday.",
        "I love how polished the presentation looked!",
    ]
    if args.encoder_type == "transformer" and tokenizer_cache_fn is not None:
        populate_tokenizer_cache(sample_inputs, "sample output set")

    fp16_warning_emitted = False

    def run_single_split(fold_index: int, train_indices: Sequence[int],
                         val_indices: Sequence[int]) -> FoldResult:
        nonlocal fp16_warning_emitted
        fold_suffix = (
            f"fold{fold_index:02d}of{total_folds:02d}"
            if total_folds > 1
            else None
        )

        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        train_weights = [1.0] * len(train_texts)
        base_train_size = len(train_texts)
        supervised_distribution = Counter(train_labels)

        val_texts = [texts[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        if not val_texts:
            raise RuntimeError("Validation split produced no examples; adjust --test-ratio or --folds.")

        val_dataset = IntentDataset(
            val_texts,
            val_labels,
            vocab=vocab,
            label_to_idx=label_to_idx,
            max_len=max_seq_len,
            tokenizer=tokenizer_obj,
            tokenizer_cache=tokenizer_cache_fn,
            embedding_model=embedding_fn,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        model = build_model()
        effective_lr = (
            args.transformer_learning_rate
            if args.encoder_type == "transformer"
            else args.learning_rate
        )
        criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=args.label_smoothing)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=effective_lr,
            weight_decay=args.weight_decay,
        )

        amp_available = GradScaler is not None
        use_amp = bool(args.fp16 and torch.cuda.is_available() and amp_available)
        if args.fp16 and not torch.cuda.is_available() and not fp16_warning_emitted:
            print("fp16 requested but CUDA is not available; training with full precision.")
            fp16_warning_emitted = True
        elif args.fp16 and not amp_available and not fp16_warning_emitted:
            print("fp16 requested but AMP utilities are unavailable; training with full precision.")
            fp16_warning_emitted = True
        scaler = create_grad_scaler(use_amp)

        unlabeled_texts = list(unlabeled_master)
        initial_unlabeled = len(unlabeled_texts)

        print(
            f"Fold {fold_index}/{total_folds}: training on {len(train_texts)} labelled examples across {num_classes} intents; "
            f"validation set has {len(val_texts)} examples."
        )
        if args.encoder_type == "transformer":
            tokenizer_size = len(tokenizer_obj) if tokenizer_obj is not None else "unknown"
            print(
                f"Fold {fold_index}/{total_folds}: transformer tokenizer '{args.transformer_model}' with vocabulary size {tokenizer_size} "
                f"and max sequence length {max_seq_len} tokens."
            )
        elif args.encoder_type == "st":
            cache_max = None
            if callable(embedding_cache_info):
                info = embedding_cache_info()
                cache_max = info.maxsize
            cache_desc = cache_max if cache_max not in (None, 0) else "unbounded"
            print(
                f"Fold {fold_index}/{total_folds}: sentence-transformer '{args.sentence_transformer_model}' embeddings (dimension {sentence_embedding_dim}, "
                f"cache size {cache_desc})."
            )
        else:
            print(
                f"Fold {fold_index}/{total_folds}: vocabulary size {len(vocab)} (min frequency = {args.min_freq}); max sequence length {max_seq_len} tokens."
            )

        history: List[Dict[str, object]] = []
        pseudo_rounds: List[Dict[str, object]] = []
        augmentation_events: List[Dict[str, object]] = []
        pseudo_examples_store: List[Tuple[str, str, float]] = []
        global_epoch = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val_acc = -float("inf")
        best_entry: Optional[Dict[str, object]] = None
        best_model_source = "model"
        epochs_since_improvement = 0
        optimizer_step_counter = 0
        ema_update_counter = 0
        swa_update_counter = 0
        total_pseudo_added = 0
        total_augmented_examples = 0

        grad_acc_steps = max(1, args.grad_accumulation_steps)
        augmentation_rng = random.Random(args.seed * 1009 + fold_index)

        ema_model: Optional[AveragedModel] = None
        if args.ema_decay > 0:
            ema_model = create_ema_model(model, args.ema_decay)
            ema_model.to(device)
        swa_model: Optional[AveragedModel] = None
        if args.swa_start_epoch > 0:
            swa_model = AveragedModel(model)
            swa_model.to(device)
        swa_scheduler_obj: Optional[SWALR] = None

        def run_stage(stage_name: str, epochs: int) -> bool:
            nonlocal global_epoch, best_state, best_val_acc, epochs_since_improvement, best_entry
            nonlocal optimizer_step_counter, ema_update_counter, swa_update_counter, best_model_source
            nonlocal total_augmented_examples, swa_scheduler_obj
            if epochs <= 0:
                return False
            augmented_texts, augmented_labels, augmented_weights, augmented_count = augment_training_corpus(
                train_texts,
                train_labels,
                train_weights,
                probability=args.augment_probability,
                strategies=augment_strategies,
                max_copies=args.augment_max_copies,
                max_transforms=args.augment_max_transforms,
                rng=augmentation_rng,
            )
            total_augmented_examples += augmented_count
            if augmented_count and args.augment_probability > 0:
                print(
                    f"Fold {fold_index}/{total_folds} – stage '{stage_name}': generated {augmented_count} augmented variants."
                )
            train_dataset = IntentDataset(
                augmented_texts,
                augmented_labels,
                vocab=vocab,
                label_to_idx=label_to_idx,
                max_len=max_seq_len,
                sample_weights=augmented_weights,
                tokenizer=tokenizer_obj,
                tokenizer_cache=tokenizer_cache_fn,
                embedding_model=embedding_fn,
            )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            effective_steps_per_epoch = math.ceil(len(train_loader) / grad_acc_steps) if len(train_loader) else 0
            scheduler, per_batch = create_scheduler(
                optimizer,
                args.scheduler,
                epochs,
                effective_steps_per_epoch,
                effective_lr,
            )

            for local_epoch in range(1, epochs + 1):
                epoch_start = time.perf_counter()
                global_epoch += 1
                ema_active = ema_model is not None and global_epoch >= args.ema_start_epoch
                swa_active = (
                    swa_model is not None
                    and args.swa_start_epoch > 0
                    and global_epoch >= args.swa_start_epoch
                )

                optimizer.grad_accumulation_steps = grad_acc_steps
                optimizer.ema_model = ema_model
                optimizer.ema_active = ema_active
                optimizer.swa_model = swa_model
                optimizer.swa_active = swa_active

                scheduler_for_epoch = scheduler if not (swa_active and args.swa_start_epoch > 0) else None
                per_batch_for_epoch = per_batch and scheduler_for_epoch is not None

                train_loss, train_acc, stats = train_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    scheduler=scheduler_for_epoch if per_batch_for_epoch else None,
                    scheduler_step_per_batch=per_batch_for_epoch,
                    scaler=scaler,
                    max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
                )

                optimizer_step_counter += stats["optimizer_steps"]
                ema_update_counter += stats["ema_updates"]
                swa_update_counter += stats["swa_updates"]

                if swa_active:
                    if swa_scheduler_obj is None:
                        swa_lr = args.swa_lr if args.swa_lr > 0 else optimizer.param_groups[0]["lr"]
                        swa_scheduler_obj = SWALR(
                            optimizer,
                            swa_lr=swa_lr,
                            anneal_epochs=max(1, args.swa_anneal_epochs),
                            anneal_strategy="cos",
                        )
                    swa_scheduler_obj.step()
                elif scheduler is not None and not per_batch:
                    scheduler.step()

                eval_model: nn.Module = model
                eval_source = "model"
                if swa_active and swa_model is not None and swa_update_counter > 0:
                    eval_model = swa_model
                    eval_source = "swa"
                if (
                    ema_model is not None
                    and ema_active
                    and args.ema_use_for_eval
                    and ema_update_counter > 0
                ):
                    eval_model = ema_model
                    eval_source = "ema"

                val_loss, val_acc = evaluate(eval_model, val_loader, criterion, device)
                current_lr = optimizer.param_groups[0]["lr"]
                history.append(
                    {
                        "epoch": float(global_epoch),
                        "stage": stage_name,
                        "train_loss": float(train_loss),
                        "train_accuracy": float(train_acc),
                        "val_loss": float(val_loss),
                        "val_accuracy": float(val_acc),
                        "learning_rate": float(current_lr),
                        "train_examples": float(len(train_dataset)),
                        "optimizer_steps": float(stats["optimizer_steps"]),
                        "ema_active": bool(ema_active),
                        "swa_active": bool(swa_active),
                        "evaluation_model": eval_source,
                        "augmented_examples": float(augmented_count),
                    }
                )

                elapsed = time.perf_counter() - epoch_start
                print(
                    f"Fold {fold_index}/{total_folds} epoch {global_epoch:03d} [{stage_name}] "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"train_acc={train_acc * 100:.2f}% val_acc={val_acc * 100:.2f}% "
                    f"lr={current_lr:.6f} ({elapsed:.1f}s)"
                )

                if val_acc > best_val_acc + 1e-6:
                    best_val_acc = val_acc
                    best_state = clone_model_state_dict(eval_model)
                    best_entry = {
                        "epoch": float(global_epoch),
                        "stage": stage_name,
                        "train_accuracy": float(train_acc),
                        "train_loss": float(train_loss),
                        "val_accuracy": float(val_acc),
                        "val_loss": float(val_loss),
                        "learning_rate": float(current_lr),
                        "evaluation_model": eval_source,
                    }
                    best_model_source = eval_source
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if args.early_stop_patience and epochs_since_improvement >= args.early_stop_patience:
                        print("Early stopping triggered due to stagnant validation performance.")
                        return True

            augmentation_events.append(
                {
                    "stage": stage_name,
                    "epochs": float(epochs),
                    "augmented_examples": float(augmented_count),
                }
            )
            if swa_model is not None and swa_update_counter > 0:
                try:
                    update_bn(train_loader, swa_model)
                except Exception as exc:  # pragma: no cover - best-effort update
                    print(f"Warning: failed to update SWA batch-norm statistics: {exc}")
            return False

        stop_training = run_stage("supervised", args.epochs)

        if not stop_training and unlabeled_texts and args.self_train_rounds > 0 and args.self_train_epochs > 0:
            for round_idx in range(1, args.self_train_rounds + 1):
                current_threshold = compute_round_threshold(
                    args.self_train_threshold,
                    round_idx,
                    args.self_train_threshold_decay,
                    args.self_train_min_threshold,
                )
                pseudo_source: nn.Module = model
                if (
                    ema_model is not None
                    and args.ema_use_for_eval
                    and ema_update_counter > 0
                ):
                    pseudo_source = ema_model
                elif swa_model is not None and swa_update_counter > 0:
                    pseudo_source = swa_model
                confident, unlabeled_texts = pseudo_label_unlabeled(
                    pseudo_source,
                    unlabeled_texts,
                    vocab=vocab,
                    label_to_idx=label_to_idx,
                    max_len=max_seq_len,
                    device=device,
                    threshold=current_threshold,
                    tokenizer=tokenizer_obj,
                    tokenizer_cache=tokenizer_cache_fn,
                    embedding_model=embedding_fn,
                )
                if not confident:
                    print(
                        f"Fold {fold_index}/{total_folds} self-training round {round_idx}: "
                        f"no predictions met the confidence threshold {current_threshold:.3f}."
                    )
                    continue
                avg_conf = sum(score for _, _, score in confident) / len(confident)
                pseudo_counts = Counter(label for _, label, _ in confident)
                print(
                    f"Fold {fold_index}/{total_folds} self-training round {round_idx}: added {len(confident)} pseudo-labelled examples "
                    f"(avg confidence {avg_conf:.3f}, threshold {current_threshold:.3f}). Remaining unlabeled: {len(unlabeled_texts)}"
                )
                added_examples = 0
                for text, label, score in confident:
                    weight = compute_pseudo_weight(
                        float(args.self_train_weight),
                        float(score),
                        float(current_threshold),
                        float(args.self_train_confidence_power),
                        float(args.self_train_max_weight_multiplier),
                    )
                    train_texts.append(text)
                    train_labels.append(label)
                    train_weights.append(weight)
                    pseudo_examples_store.append((text, label, weight))
                    added_examples += 1
                total_pseudo_added += added_examples
                pseudo_rounds.append(
                    {
                        "round": float(round_idx),
                        "added_examples": float(added_examples),
                        "average_confidence": float(avg_conf),
                        "threshold": float(current_threshold),
                        "label_histogram": dict(sorted(pseudo_counts.items())),
                    }
                )
                epochs_since_improvement = 0
                stop_training = run_stage(f"self-train-{round_idx}", args.self_train_epochs)
                if stop_training:
                    break

        if best_state is None:
            reference_model: nn.Module = model
            if (
                ema_model is not None
                and args.ema_use_for_eval
                and ema_update_counter > 0
            ):
                reference_model = ema_model
                best_model_source = "ema"
            elif swa_model is not None and swa_update_counter > 0:
                reference_model = swa_model
                best_model_source = "swa"
            best_state = clone_model_state_dict(reference_model)

        model.load_state_dict(best_state)
        model.to(device)

        val_loss_final, val_acc_final, val_targets, val_predictions, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            return_details=True,
        )
        class_metrics = compute_classification_metrics(val_targets, val_predictions, label_to_idx=label_to_idx)

        if best_entry is None:
            best_entry = history[-1] if history else {
                "epoch": 0.0,
                "stage": "supervised",
                "train_accuracy": 0.0,
                "train_loss": 0.0,
                "val_accuracy": best_val_acc,
                "val_loss": val_loss_final,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "evaluation_model": best_model_source,
            }

        metadata = {
            "encoder_type": args.encoder_type,
            "model_name": args.model_name,
            "trainer_version": TRAINER_VERSION,
            "dataset_path": str(args.dataset),
            "dataset_checksum": dataset_checksum,
            "dataset_examples": len(texts),
            "num_labels": num_classes,
            "vocab": vocab if args.encoder_type == "bilstm" else None,
            "label_to_idx": label_to_idx,
            "max_seq_len": max_seq_len,
            "embedding_dim": (
                args.embedding_dim
                if args.encoder_type == "bilstm"
                else (sentence_embedding_dim if args.encoder_type == "st" else None)
            ),
            "hidden_dim": args.hidden_dim if args.encoder_type == "bilstm" else None,
            "ffn_dim": args.ffn_dim if args.encoder_type == "bilstm" else None,
            "encoder_layers": args.encoder_layers if args.encoder_type == "bilstm" else None,
            "attention_heads": args.attention_heads if args.encoder_type == "bilstm" else None,
            "dropout": args.dropout if args.encoder_type == "bilstm" else None,
            "test_ratio": args.test_ratio if total_folds == 1 else None,
            "seed": args.seed,
            "scheduler": args.scheduler,
            "learning_rate": effective_lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "label_smoothing": args.label_smoothing,
            "use_fp16": bool(use_amp),
            "training_examples_supervised": base_train_size,
            "training_examples_final": len(train_texts),
            "validation_examples": len(val_texts),
            "promotion_tolerance": args.promotion_tolerance,
            "class_distribution_total": dict(sorted(Counter(labels).items())),
            "class_distribution_supervised": dict(sorted(supervised_distribution.items())),
            "class_distribution_final": dict(sorted(Counter(train_labels).items())),
            "validation_distribution": dict(sorted(Counter(val_labels).items())),
            "self_training": {
                "rounds_configured": args.self_train_rounds,
                "rounds_ran": len(pseudo_rounds),
                "epochs_per_round": args.self_train_epochs,
                "confidence_threshold": args.self_train_threshold,
                "threshold_decay": args.self_train_threshold_decay,
                "min_threshold": args.self_train_min_threshold,
                "pseudo_example_weight": args.self_train_weight,
                "confidence_power": args.self_train_confidence_power,
                "max_weight_multiplier": args.self_train_max_weight_multiplier,
                "examples_added": total_pseudo_added,
                "round_details": pseudo_rounds,
                "initial_unlabeled": initial_unlabeled,
                "remaining_unlabeled": len(unlabeled_texts),
            },
            "training_history": history,
            "augmentation": {
                "probability": args.augment_probability,
                "strategies": augment_strategies,
                "max_copies": args.augment_max_copies,
                "max_transforms": args.augment_max_transforms,
                "total_generated": total_augmented_examples,
                "events": augmentation_events,
            },
            "advanced_training": {
                "gradient_accumulation_steps": grad_acc_steps,
                "ema_decay": args.ema_decay,
                "ema_start_epoch": args.ema_start_epoch,
                "ema_updates": ema_update_counter,
                "ema_used_for_evaluation": args.ema_use_for_eval,
                "swa_start_epoch": args.swa_start_epoch,
                "swa_lr": args.swa_lr if args.swa_lr > 0 else effective_lr,
                "swa_anneal_epochs": args.swa_anneal_epochs,
                "swa_updates": swa_update_counter,
                "optimizer_steps": optimizer_step_counter,
                "best_model_source": best_model_source,
            },
            "validation_report": class_metrics,
            "best_val_accuracy": best_val_acc,
            "fold_index": fold_index,
            "folds": total_folds,
            "folds_requested": folds_requested,
        }
        if args.unlabeled_dataset:
            metadata["unlabeled_dataset"] = str(args.unlabeled_dataset)
            metadata["unlabeled_checksum"] = unlabeled_checksum
        if args.encoder_type == "transformer":
            metadata["transformer_model"] = args.transformer_model
            if tokenizer_obj is not None:
                metadata["tokenizer_name_or_path"] = getattr(tokenizer_obj, "name_or_path", args.transformer_model)
                metadata["tokenizer_vocab_size"] = len(tokenizer_obj)
        if args.encoder_type == "st":
            metadata["sentence_transformer_model"] = args.sentence_transformer_model
            metadata["sentence_transformer_hidden_dim"] = args.st_hidden_dim
            metadata["sentence_transformer_dropout"] = args.st_dropout

        metrics: Dict[str, object] = {
            "model_name": args.model_name,
            "trainer_version": TRAINER_VERSION,
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "validation_accuracy": float(best_val_acc),
            "train_accuracy_at_best": float(best_entry["train_accuracy"]),
            "best_epoch": float(best_entry["epoch"]),
            "best_stage": best_entry["stage"],
            "epochs_ran": float(global_epoch),
            "dataset_examples": len(texts),
            "dataset_checksum": dataset_checksum,
            "num_labels": num_classes,
            "pseudo_examples_added": total_pseudo_added,
            "remaining_unlabeled": len(unlabeled_texts),
            "self_training_rounds_configured": args.self_train_rounds,
            "self_training_rounds_completed": len(pseudo_rounds),
            "promotion_tolerance": args.promotion_tolerance,
            "encoder_type": args.encoder_type,
            "effective_learning_rate": effective_lr,
            "fold_index": fold_index,
            "folds": total_folds,
            "optimizer_steps": float(optimizer_step_counter),
            "ema_updates": float(ema_update_counter),
            "swa_updates": float(swa_update_counter),
            "augmentation_generated": float(total_augmented_examples),
            "validation_macro_f1": float(class_metrics["macro_f1"]),
            "validation_weighted_f1": float(class_metrics["weighted_f1"]),
            "validation_macro_precision": float(class_metrics["macro_precision"]),
            "validation_macro_recall": float(class_metrics["macro_recall"]),
            "best_model_source": best_model_source,
        }
        if unlabeled_checksum is not None:
            metrics["unlabeled_checksum"] = unlabeled_checksum
        if args.unlabeled_dataset:
            metrics["unlabeled_dataset"] = str(args.unlabeled_dataset)
        if args.encoder_type == "transformer":
            metrics["transformer_model"] = args.transformer_model
            if tokenizer_obj is not None:
                metrics["tokenizer_vocab_size"] = len(tokenizer_obj)
        if args.encoder_type == "st":
            metrics["sentence_transformer_model"] = args.sentence_transformer_model
            metrics["sentence_transformer_hidden_dim"] = args.st_hidden_dim
            metrics["sentence_transformer_dropout"] = args.st_dropout

        sample_outputs: List[Dict[str, str]] = []
        for sample in sample_inputs:
            predicted_label = predict_label(
                model,
                sample,
                vocab=vocab,
                label_to_idx=label_to_idx,
                max_len=max_seq_len,
                device=device,
                tokenizer=tokenizer_obj,
                tokenizer_cache=tokenizer_cache_fn,
                embedding_model=embedding_fn,
            )
            response = generate_response(predicted_label, sample)
            sample_outputs.append(
                {
                    "input": sample,
                    "predicted_label": predicted_label,
                    "response": response,
                }
            )

        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return FoldResult(
            fold_index=fold_index,
            val_accuracy=float(best_val_acc),
            train_accuracy_at_best=float(best_entry["train_accuracy"]),
            metrics=metrics,
            metadata=metadata,
            history=history,
            pseudo_rounds=pseudo_rounds,
            total_pseudo_added=total_pseudo_added,
            pseudo_examples=pseudo_examples_store,
            model_state=best_state,
            sample_outputs=sample_outputs,
            run_tag_suffix=fold_suffix,
        )
    def run_full_dataset_training(best_fold: FoldResult) -> Optional[Dict[str, object]]:
        nonlocal fp16_warning_emitted
        if args.final_train_epochs <= 0:
            return None

        final_texts = list(texts)
        final_labels = list(labels)
        final_weights = [1.0] * len(final_texts)
        pseudo_total = 0
        if args.final_use_pseudo:
            pseudo_cache: Dict[str, Tuple[str, float]] = {}
            for result in fold_results:
                for text, label, weight in result.pseudo_examples:
                    existing = pseudo_cache.get(text)
                    if existing is None or existing[1] < weight:
                        pseudo_cache[text] = (label, weight)
            for text, (label, weight) in pseudo_cache.items():
                final_texts.append(text)
                final_labels.append(label)
                final_weights.append(weight)
            pseudo_total = len(pseudo_cache)
        print(
            f"Final full-data training: {len(final_texts)} supervised examples ({len(texts)} labelled + {pseudo_total} pseudo-labelled)."
        )

        final_batch_size = args.final_train_batch_size if args.final_train_batch_size > 0 else args.batch_size
        base_effective_lr = args.transformer_learning_rate if args.encoder_type == "transformer" else args.learning_rate
        final_lr = args.final_train_learning_rate if args.final_train_learning_rate > 0 else base_effective_lr
        final_weight_decay = args.final_train_weight_decay if args.final_train_weight_decay >= 0 else args.weight_decay
        final_scheduler_choice = args.final_train_scheduler if args.final_train_scheduler != "inherit" else args.scheduler

        final_model = build_model()
        final_model.load_state_dict(best_fold.model_state)
        final_model.to(device)

        amp_available = GradScaler is not None
        use_amp = bool(args.fp16 and torch.cuda.is_available() and amp_available)
        if args.fp16 and not torch.cuda.is_available() and not fp16_warning_emitted:
            print("fp16 requested but CUDA is not available; training with full precision for final stage.")
            fp16_warning_emitted = True
        elif args.fp16 and not amp_available and not fp16_warning_emitted:
            print("fp16 requested but AMP utilities are unavailable; training with full precision for final stage.")
            fp16_warning_emitted = True
        final_scaler = create_grad_scaler(use_amp)

        criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=args.label_smoothing)
        optimizer = torch.optim.AdamW(
            final_model.parameters(),
            lr=final_lr,
            weight_decay=final_weight_decay,
        )

        grad_acc_steps = max(1, args.grad_accumulation_steps)
        augmentation_rng = random.Random(args.seed * 173 + 2048)
        ema_model: Optional[AveragedModel] = None
        if args.ema_decay > 0:
            ema_model = create_ema_model(final_model, args.ema_decay)
            ema_model.to(device)
        swa_model: Optional[AveragedModel] = None
        if args.swa_start_epoch > 0:
            swa_model = AveragedModel(final_model)
            swa_model.to(device)
        swa_scheduler: Optional[SWALR] = None

        needs_teacher = args.distill_epochs > 0 or args.distill_keep_during_final
        distillation_stats: Dict[str, object] = {
            "enabled": bool(needs_teacher),
            "epochs": int(max(args.distill_epochs, 0)),
            "alpha": float(args.distill_alpha),
            "temperature": float(args.distill_temperature),
            "min_confidence": float(args.distill_min_confidence),
            "confidence_power": float(args.distill_confidence_power),
            "max_weight_multiplier": float(args.distill_max_weight_multiplier),
            "keep_during_final": bool(args.distill_keep_during_final),
            "teachers_considered": 0,
            "teacher_indices": [],
            "teacher_pool_mean_accuracy": 0.0,
            "teacher_coverage": 0.0,
            "teacher_active_examples": 0,
            "teacher_mean_confidence": 0.0,
            "teacher_median_confidence": 0.0,
            "teacher_min_confidence": 0.0,
            "teacher_max_confidence": 0.0,
            "teacher_confidence_std": 0.0,
        }
        distillation_logits: Optional[List[Optional[List[float]]]] = None
        distillation_weights: Optional[List[float]] = None
        distillation_config: Optional[DistillationConfig] = None
        teacher_logits_for_final: Optional[List[Optional[List[float]]]] = None
        if needs_teacher:
            teacher_pool = sorted(fold_results, key=lambda res: res.val_accuracy, reverse=True)
            if args.distill_max_teachers > 0:
                teacher_pool = teacher_pool[: args.distill_max_teachers]
            distillation_stats["teachers_considered"] = len(teacher_pool)
            distillation_stats["teacher_indices"] = [result.fold_index for result in teacher_pool]
            if teacher_pool:
                pool_mean_accuracy = mean(result.val_accuracy for result in teacher_pool)
                distillation_stats["teacher_pool_mean_accuracy"] = float(pool_mean_accuracy)
                teacher_dataset = IntentDataset(
                    final_texts,
                    final_labels,
                    vocab=vocab,
                    label_to_idx=label_to_idx,
                    max_len=max_seq_len,
                    sample_weights=final_weights,
                    tokenizer=tokenizer_obj,
                    tokenizer_cache=tokenizer_cache_fn,
                    embedding_model=embedding_fn,
                )
                teacher_loader = DataLoader(teacher_dataset, batch_size=final_batch_size, shuffle=False)
                aggregated_logits: Optional[List[List[float]]] = None
                for teacher_result in teacher_pool:
                    teacher_model = build_model()
                    teacher_model.load_state_dict(teacher_result.model_state)
                    teacher_model.to(device)
                    teacher_model.eval()
                    collected_logits: List[List[float]] = []
                    with torch.no_grad():
                        for batch in teacher_loader:
                            if len(batch) == 5:
                                inputs, _labels, _weights, attention_mask, _teacher_logits = batch
                            else:
                                inputs, _labels, _weights, attention_mask = batch  # type: ignore[misc]
                            inputs = inputs.to(device)
                            attention_mask = attention_mask.to(device)
                            outputs = teacher_model(inputs, attention_mask=attention_mask)
                            collected_logits.extend(outputs.detach().cpu().tolist())
                    if aggregated_logits is None:
                        aggregated_logits = [list(map(float, row)) for row in collected_logits]
                    else:
                        for idx, row in enumerate(collected_logits):
                            accumulator = aggregated_logits[idx]
                            for col, value in enumerate(row):
                                accumulator[col] += float(value)
                    teacher_model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if aggregated_logits is not None:
                    teacher_count = max(1, len(teacher_pool))
                    averaged_tensor = torch.tensor(aggregated_logits, dtype=torch.float32) / float(teacher_count)
                    probability_tensor = torch.softmax(averaged_tensor, dim=1)
                    confidence_values = probability_tensor.max(dim=1).values.tolist()
                    averaged_logits = averaged_tensor.tolist()
                    distillation_logits = []
                    distillation_weights = []
                    active_confidences: List[float] = []
                    for row, confidence in zip(averaged_logits, confidence_values):
                        if confidence >= args.distill_min_confidence:
                            distillation_logits.append([float(x) for x in row])
                            scaled_weight = scale_confidence_weight(
                                1.0,
                                float(confidence),
                                max(args.distill_min_confidence, 1e-6),
                                float(args.distill_confidence_power),
                                float(args.distill_max_weight_multiplier) if args.distill_max_weight_multiplier > 0 else float("inf"),
                            )
                            distillation_weights.append(float(scaled_weight))
                            active_confidences.append(float(confidence))
                        else:
                            distillation_logits.append(None)
                            distillation_weights.append(0.0)
                    coverage = sum(1 for entry in distillation_logits if entry is not None)
                    coverage_ratio = coverage / max(1, len(distillation_logits))
                    distillation_stats.update(
                        {
                            "teacher_coverage": float(coverage_ratio),
                            "teacher_active_examples": int(coverage),
                            "teacher_mean_confidence": float(mean(active_confidences)) if active_confidences else 0.0,
                            "teacher_median_confidence": float(median(active_confidences)) if active_confidences else 0.0,
                            "teacher_min_confidence": float(min(active_confidences)) if active_confidences else 0.0,
                            "teacher_max_confidence": float(max(active_confidences)) if active_confidences else 0.0,
                            "teacher_confidence_std": float(pstdev(active_confidences)) if len(active_confidences) > 1 else 0.0,
                        }
                    )
                    if coverage > 0:
                        distillation_config = DistillationConfig(
                            alpha=float(args.distill_alpha),
                            temperature=float(args.distill_temperature),
                        )
                        teacher_logits_for_final = list(distillation_logits)
                else:
                    distillation_stats.update(
                        {
                            "teacher_coverage": 0.0,
                            "teacher_active_examples": 0,
                            "teacher_mean_confidence": 0.0,
                            "teacher_median_confidence": 0.0,
                            "teacher_min_confidence": 0.0,
                            "teacher_max_confidence": 0.0,
                            "teacher_confidence_std": 0.0,
                        }
                    )
            else:
                distillation_stats.update(
                    {
                        "teacher_pool_mean_accuracy": 0.0,
                        "teacher_coverage": 0.0,
                        "teacher_active_examples": 0,
                        "teacher_mean_confidence": 0.0,
                        "teacher_median_confidence": 0.0,
                        "teacher_min_confidence": 0.0,
                        "teacher_max_confidence": 0.0,
                        "teacher_confidence_std": 0.0,
                    }
                )

        augmented_texts, augmented_labels, augmented_weights, augmented_count = augment_training_corpus(
            final_texts,
            final_labels,
            final_weights,
            probability=args.augment_probability,
            strategies=augment_strategies,
            max_copies=args.augment_max_copies,
            max_transforms=args.augment_max_transforms,
            rng=augmentation_rng,
        )
        teacher_logits_for_final_dataset: Optional[List[Optional[List[float]]]] = None
        if (
            teacher_logits_for_final is not None
            and distillation_config is not None
            and args.distill_keep_during_final
        ):
            teacher_logits_for_final_dataset = list(teacher_logits_for_final)
            if len(teacher_logits_for_final_dataset) < len(augmented_texts):
                teacher_logits_for_final_dataset.extend(
                    [None] * (len(augmented_texts) - len(teacher_logits_for_final_dataset))
                )

        train_dataset = IntentDataset(
            augmented_texts,
            augmented_labels,
            vocab=vocab,
            label_to_idx=label_to_idx,
            max_len=max_seq_len,
            sample_weights=augmented_weights,
            tokenizer=tokenizer_obj,
            tokenizer_cache=tokenizer_cache_fn,
            embedding_model=embedding_fn,
            teacher_logits=teacher_logits_for_final_dataset,
        )
        train_loader = DataLoader(train_dataset, batch_size=final_batch_size, shuffle=True)
        eval_loader = DataLoader(train_dataset, batch_size=final_batch_size)
        effective_steps_per_epoch = math.ceil(len(train_loader) / grad_acc_steps) if len(train_loader) else 0
        scheduler, per_batch = create_scheduler(
            optimizer,
            final_scheduler_choice,
            args.final_train_epochs,
            effective_steps_per_epoch,
            final_lr,
        )

        history: List[Dict[str, object]] = []
        optimizer_steps_total = 0
        ema_updates_total = 0
        swa_updates_total = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_entry: Optional[Dict[str, object]] = None
        best_model_source = "model"
        best_accuracy = -float("inf")
        epochs_since_improvement = 0
        total_epochs = 0
        final_stage_distillation = (
            distillation_config if args.distill_keep_during_final and distillation_config is not None else None
        )
        distill_epochs_completed = 0
        if args.distill_epochs > 0 and distillation_config is not None and distillation_logits is not None:
            distill_dataset = IntentDataset(
                final_texts,
                final_labels,
                vocab=vocab,
                label_to_idx=label_to_idx,
                max_len=max_seq_len,
                sample_weights=distillation_weights if distillation_weights is not None else final_weights,
                tokenizer=tokenizer_obj,
                tokenizer_cache=tokenizer_cache_fn,
                embedding_model=embedding_fn,
                teacher_logits=distillation_logits,
            )
            distill_loader = DataLoader(distill_dataset, batch_size=final_batch_size, shuffle=True)
            distill_eval_loader = DataLoader(distill_dataset, batch_size=final_batch_size)
            for epoch in range(1, args.distill_epochs + 1):
                epoch_start = time.perf_counter()
                total_epochs += 1
                ema_active = ema_model is not None and total_epochs >= args.ema_start_epoch
                swa_active = (
                    swa_model is not None
                    and args.swa_start_epoch > 0
                    and total_epochs >= args.swa_start_epoch
                )

                optimizer.grad_accumulation_steps = grad_acc_steps
                optimizer.ema_model = ema_model
                optimizer.ema_active = ema_active
                optimizer.swa_model = swa_model
                optimizer.swa_active = swa_active

                train_loss, train_acc, stats = train_epoch(
                    final_model,
                    distill_loader,
                    criterion,
                    optimizer,
                    device,
                    scaler=final_scaler,
                    max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
                    distillation_config=distillation_config,
                )
                optimizer_steps_total += stats["optimizer_steps"]
                ema_updates_total += stats["ema_updates"]
                swa_updates_total += stats["swa_updates"]

                eval_model: nn.Module = final_model
                eval_source = "model"
                if swa_active and swa_model is not None and swa_updates_total > 0:
                    eval_model = swa_model
                    eval_source = "swa"
                if (
                    ema_model is not None
                    and ema_active
                    and args.ema_use_for_eval
                    and ema_updates_total > 0
                ):
                    eval_model = ema_model
                    eval_source = "ema"

                eval_loss, eval_acc, eval_targets, eval_predictions, _ = evaluate(
                    eval_model,
                    distill_eval_loader,
                    criterion,
                    device,
                    return_details=True,
                )
                current_lr = optimizer.param_groups[0]["lr"]
                history.append(
                    {
                        "epoch": float(total_epochs),
                        "stage": "distill",
                        "train_loss": float(train_loss),
                        "train_accuracy": float(train_acc),
                        "val_loss": float(eval_loss),
                        "val_accuracy": float(eval_acc),
                        "learning_rate": float(current_lr),
                        "train_examples": float(len(distill_dataset)),
                        "optimizer_steps": float(stats["optimizer_steps"]),
                        "ema_active": bool(ema_active),
                        "swa_active": bool(swa_active),
                        "evaluation_model": eval_source,
                        "teacher_examples": float(distillation_stats.get("teacher_active_examples", 0)),
                        "teacher_coverage": float(distillation_stats.get("teacher_coverage", 0.0)),
                    }
                )
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"Final stage distill epoch {total_epochs:03d} "
                    f"train_loss={train_loss:.4f} val_loss={eval_loss:.4f} "
                    f"train_acc={train_acc * 100:.2f}% val_acc={eval_acc * 100:.2f}% "
                    f"lr={current_lr:.6f} ({elapsed:.1f}s)"
                )
                if eval_acc > best_accuracy + 1e-6:
                    best_accuracy = eval_acc
                    best_state = clone_model_state_dict(eval_model)
                    best_entry = {
                        "epoch": float(total_epochs),
                        "stage": "distill",
                        "train_accuracy": float(train_acc),
                        "train_loss": float(train_loss),
                        "val_accuracy": float(eval_acc),
                        "val_loss": float(eval_loss),
                        "learning_rate": float(current_lr),
                        "evaluation_model": eval_source,
                    }
                    best_model_source = eval_source
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if args.early_stop_patience and epochs_since_improvement >= args.early_stop_patience:
                        print("Distillation early stopping triggered due to stagnant performance.")
                        break
            distill_epochs_completed = total_epochs
            epochs_since_improvement = 0
        else:
            distillation_stats.setdefault("teacher_coverage", 0.0)
            distillation_stats.setdefault("teacher_active_examples", 0)

        for epoch in range(1, args.final_train_epochs + 1):
            epoch_start = time.perf_counter()
            total_epochs += 1
            ema_active = ema_model is not None and total_epochs >= args.ema_start_epoch
            swa_active = swa_model is not None and args.swa_start_epoch > 0 and total_epochs >= args.swa_start_epoch

            optimizer.grad_accumulation_steps = grad_acc_steps
            optimizer.ema_model = ema_model
            optimizer.ema_active = ema_active
            optimizer.swa_model = swa_model
            optimizer.swa_active = swa_active

            scheduler_for_epoch = scheduler if not (swa_active and args.swa_start_epoch > 0) else None
            per_batch_for_epoch = per_batch and scheduler_for_epoch is not None

            train_loss, train_acc, stats = train_epoch(
                final_model,
                train_loader,
                criterion,
                optimizer,
                device,
                scheduler=scheduler_for_epoch if per_batch_for_epoch else None,
                scheduler_step_per_batch=per_batch_for_epoch,
                scaler=final_scaler,
                max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
                distillation_config=final_stage_distillation,
            )
            optimizer_steps_total += stats["optimizer_steps"]
            ema_updates_total += stats["ema_updates"]
            swa_updates_total += stats["swa_updates"]

            if swa_active:
                if swa_scheduler is None:
                    swa_lr = args.swa_lr if args.swa_lr > 0 else optimizer.param_groups[0]["lr"]
                    swa_scheduler = SWALR(
                        optimizer,
                        swa_lr=swa_lr,
                        anneal_epochs=max(1, args.swa_anneal_epochs),
                        anneal_strategy="cos",
                    )
                swa_scheduler.step()
            elif scheduler is not None and not per_batch:
                scheduler.step()

            eval_model: nn.Module = final_model
            eval_source = "model"
            if swa_active and swa_model is not None and swa_updates_total > 0:
                eval_model = swa_model
                eval_source = "swa"
            if (
                ema_model is not None
                and ema_active
                and args.ema_use_for_eval
                and ema_updates_total > 0
            ):
                eval_model = ema_model
                eval_source = "ema"

            eval_loss, eval_acc, eval_targets, eval_predictions, _ = evaluate(
                eval_model,
                eval_loader,
                criterion,
                device,
                return_details=True,
            )
            eval_metrics = compute_classification_metrics(
                eval_targets,
                eval_predictions,
                label_to_idx=label_to_idx,
            )
            current_lr = optimizer.param_groups[0]["lr"]
            history.append(
                {
                    "epoch": float(total_epochs),
                    "stage": "final_full",
                    "train_loss": float(train_loss),
                    "train_accuracy": float(train_acc),
                    "val_loss": float(eval_loss),
                    "val_accuracy": float(eval_acc),
                    "learning_rate": float(current_lr),
                    "train_examples": float(len(train_dataset)),
                    "optimizer_steps": float(stats["optimizer_steps"]),
                    "ema_active": bool(ema_active),
                    "swa_active": bool(swa_active),
                    "evaluation_model": eval_source,
                    "augmented_examples": float(augmented_count),
                    "teacher_examples": float(distillation_stats.get("teacher_active_examples", 0)) if final_stage_distillation is not None else 0.0,
                    "teacher_coverage": float(distillation_stats.get("teacher_coverage", 0.0)) if final_stage_distillation is not None else 0.0,
                }
            )
            elapsed = time.perf_counter() - epoch_start
            print(
                f"Final training epoch {total_epochs:03d} "
                f"train_loss={train_loss:.4f} val_loss={eval_loss:.4f} "
                f"train_acc={train_acc * 100:.2f}% val_acc={eval_acc * 100:.2f}% "
                f"lr={current_lr:.6f} ({elapsed:.1f}s)"
            )
            if eval_acc > best_accuracy + 1e-6:
                best_accuracy = eval_acc
                best_state = clone_model_state_dict(eval_model)
                best_entry = {
                    "epoch": float(total_epochs),
                    "stage": "final_full",
                    "train_accuracy": float(train_acc),
                    "train_loss": float(train_loss),
                    "val_accuracy": float(eval_acc),
                    "val_loss": float(eval_loss),
                    "learning_rate": float(current_lr),
                    "evaluation_model": eval_source,
                }
                best_model_source = eval_source
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if args.early_stop_patience and epochs_since_improvement >= args.early_stop_patience:
                    print("Final training early stopping triggered due to stagnant self-evaluation performance.")
                    break

        if swa_model is not None and swa_updates_total > 0:
            try:
                update_bn(train_loader, swa_model)
            except Exception as exc:
                print(f"Warning: failed to refresh SWA batch-norm statistics during final training: {exc}")

        if best_state is None:
            reference_model = final_model
            if (
                ema_model is not None
                and args.ema_use_for_eval
                and ema_updates_total > 0
            ):
                reference_model = ema_model
                best_model_source = "ema"
            elif swa_model is not None and swa_updates_total > 0:
                reference_model = swa_model
                best_model_source = "swa"
            best_state = clone_model_state_dict(reference_model)
            best_accuracy = eval_acc
            best_entry = history[-1] if history else {
                "epoch": float(total_epochs),
                "stage": "final_full",
                "train_accuracy": float(train_acc),
                "train_loss": float(train_loss),
                "val_accuracy": float(eval_acc),
                "val_loss": float(eval_loss),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "evaluation_model": best_model_source,
            }

        distillation_stats.setdefault("teacher_indices", distillation_stats.get("teacher_indices", []))
        distillation_stats.setdefault("teacher_pool_mean_accuracy", float(distillation_stats.get("teacher_pool_mean_accuracy", 0.0)))
        distillation_stats.setdefault("teacher_coverage", 0.0)
        distillation_stats.setdefault("teacher_active_examples", 0)
        distillation_stats["epochs_completed"] = int(distill_epochs_completed)
        distillation_stats["applied_during_final"] = bool(final_stage_distillation is not None)

        final_model.load_state_dict(best_state)
        final_model.to(device)
        final_loss, final_acc, final_targets, final_predictions, _ = evaluate(
            final_model,
            eval_loader,
            criterion,
            device,
            return_details=True,
        )
        final_metrics = compute_classification_metrics(
            final_targets,
            final_predictions,
            label_to_idx=label_to_idx,
        )

        metadata = {
            "stage": "final_full",
            "base_fold_accuracy": best_fold.val_accuracy,
            "trainer_version": TRAINER_VERSION,
            "model_name": args.model_name,
            "encoder_type": args.encoder_type,
            "dataset_path": str(args.dataset),
            "dataset_checksum": dataset_checksum,
            "dataset_examples": len(final_texts),
            "num_labels": len(label_to_idx),
            "label_to_idx": label_to_idx,
            "max_seq_len": max_seq_len,
            "pseudo_examples_used": pseudo_total,
            "training_history": history,
            "augmentation": {
                "probability": args.augment_probability,
                "strategies": augment_strategies,
                "max_copies": args.augment_max_copies,
                "max_transforms": args.augment_max_transforms,
                "total_generated": augmented_count,
            },
            "advanced_training": {
                "gradient_accumulation_steps": grad_acc_steps,
                "ema_decay": args.ema_decay,
                "ema_start_epoch": args.ema_start_epoch,
                "ema_updates": ema_updates_total,
                "ema_used_for_evaluation": args.ema_use_for_eval,
                "swa_start_epoch": args.swa_start_epoch,
                "swa_lr": args.swa_lr if args.swa_lr > 0 else final_lr,
                "swa_anneal_epochs": args.swa_anneal_epochs,
                "swa_updates": swa_updates_total,
                "optimizer_steps": optimizer_steps_total,
                "best_model_source": best_model_source,
                "learning_rate": final_lr,
                "weight_decay": final_weight_decay,
                "scheduler": final_scheduler_choice,
            },
            "validation_report": final_metrics,
            "best_epoch": best_entry["epoch"],
            "best_model_source": best_model_source,
            "distillation": distillation_stats,
        }
        if args.encoder_type == "bilstm":
            metadata["vocab"] = vocab
        if args.encoder_type == "transformer":
            metadata["transformer_model"] = args.transformer_model
            if tokenizer_obj is not None:
                metadata["tokenizer_vocab_size"] = len(tokenizer_obj)
        if args.encoder_type == "st":
            metadata["sentence_transformer_model"] = args.sentence_transformer_model
            metadata["sentence_transformer_hidden_dim"] = args.st_hidden_dim
            metadata["sentence_transformer_dropout"] = args.st_dropout

        metrics = {
            "model_name": args.model_name,
            "trainer_version": TRAINER_VERSION,
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset_examples": len(final_texts),
            "dataset_checksum": dataset_checksum,
            "num_labels": len(label_to_idx),
            "encoder_type": args.encoder_type,
            "training_accuracy": float(final_acc),
            "training_loss": float(final_loss),
            "training_macro_f1": float(final_metrics["macro_f1"]),
            "training_weighted_f1": float(final_metrics["weighted_f1"]),
            "epochs_ran": float(len(history)),
            "optimizer_steps": float(optimizer_steps_total),
            "ema_updates": float(ema_updates_total),
            "swa_updates": float(swa_updates_total),
            "pseudo_examples_used": pseudo_total,
            "augmented_examples": float(augmented_count),
            "best_epoch": float(best_entry["epoch"]),
            "best_train_accuracy": float(best_entry["val_accuracy"]),
            "best_train_loss": float(best_entry["val_loss"]),
            "best_model_source": best_model_source,
            "learning_rate": final_lr,
            "weight_decay": final_weight_decay,
            "scheduler": final_scheduler_choice,
            "distillation_epochs": int(max(args.distill_epochs, 0)),
            "distillation_epochs_completed": int(distill_epochs_completed),
            "distillation_teacher_coverage": float(distillation_stats.get("teacher_coverage", 0.0)),
            "distillation_teachers_considered": int(distillation_stats.get("teachers_considered", 0)),
            "distillation_keep_during_final": bool(args.distill_keep_during_final),
            "distillation_alpha": float(args.distill_alpha),
            "distillation_temperature": float(args.distill_temperature),
            "distillation_teacher_pool_mean_accuracy": float(distillation_stats.get("teacher_pool_mean_accuracy", 0.0)),
        }
        if args.encoder_type == "transformer":
            metrics["transformer_model"] = args.transformer_model
            if tokenizer_obj is not None:
                metrics["tokenizer_vocab_size"] = len(tokenizer_obj)
        if args.encoder_type == "st":
            metrics["sentence_transformer_model"] = args.sentence_transformer_model
            metrics["sentence_transformer_hidden_dim"] = args.st_hidden_dim
            metrics["sentence_transformer_dropout"] = args.st_dropout

        sample_outputs: List[Dict[str, str]] = []
        for sample in sample_inputs:
            predicted_label = predict_label(
                final_model,
                sample,
                vocab=vocab,
                label_to_idx=label_to_idx,
                max_len=max_seq_len,
                device=device,
                tokenizer=tokenizer_obj,
                tokenizer_cache=tokenizer_cache_fn,
                embedding_model=embedding_fn,
            )
            response = generate_response(predicted_label, sample)
            sample_outputs.append(
                {
                    "input": sample,
                    "predicted_label": predicted_label,
                    "response": response,
                }
            )

        final_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "metadata": metadata,
            "metrics": metrics,
            "model_state": best_state,
            "sample_outputs": sample_outputs,
            "training_history": history,
            "accuracy": float(final_acc),
        }

    fold_results: List[FoldResult] = []
    for fold_idx, (train_indices, val_indices) in enumerate(fold_pairs, start=1):
        fold_results.append(run_single_split(fold_idx, train_indices, val_indices))

    if not fold_results:
        raise RuntimeError("No folds were produced for training.")

    fold_accuracies = [result.val_accuracy for result in fold_results]
    cv_mean = mean(fold_accuracies)
    cv_std = pstdev(fold_accuracies) if len(fold_accuracies) > 1 else 0.0
    best_fold = max(fold_results, key=lambda result: result.val_accuracy)

    registry = ModelRegistry(args.model_dir, args.model_name, tolerance=args.promotion_tolerance)
    previous_best = registry.best_accuracy()
    promote = registry.is_improvement(best_fold.val_accuracy)

    run_dirs: Dict[int, Path] = {}
    for result in fold_results:
        fold_tag = result.run_tag_suffix or (
            f"fold{result.fold_index:02d}"
            if len(fold_results) > 1
            else None
        )
        tag_parts = [args.run_tag, fold_tag]
        run_tag = "__".join([part for part in tag_parts if part]) or None

        result.metrics["previous_best_accuracy"] = previous_best
        result.metrics["promoted_to_orion"] = bool(promote and result is best_fold)
        result.metrics["cross_validation_mean_accuracy"] = cv_mean
        result.metrics["cross_validation_std_accuracy"] = cv_std
        result.metrics["validation_accuracy_source"] = (
            "cross_validation" if len(fold_results) > 1 else "holdout"
        )
        if run_tag is not None:
            result.metrics["run_tag"] = run_tag
        elif args.run_tag is not None:
            result.metrics["run_tag"] = args.run_tag

        result.metadata["run_tag"] = run_tag if run_tag is not None else args.run_tag
        result.metadata["cross_validation"] = {
            "enabled": len(fold_results) > 1,
            "fold_index": result.fold_index,
            "folds": len(fold_results),
            "mean_validation_accuracy": cv_mean,
            "std_validation_accuracy": cv_std,
        }

        run_dir = registry.create_run_directory(result.val_accuracy, run_tag)
        model_to_save = build_model(target_device=torch.device("cpu"))
        model_to_save.load_state_dict(result.model_state)
        save_run_artifacts(
            model_to_save,
            result.metadata,
            result.metrics,
            run_dir,
            model_name=args.model_name,
            tolerance=args.promotion_tolerance,
        )
        run_dirs[result.fold_index] = run_dir

    best_run_dir = run_dirs[best_fold.fold_index]
    promotion_message_emitted = False
    if promote:
        registry.promote(best_run_dir)
        print(
            f"Promoted run to {registry.best_dir} (validation accuracy {best_fold.val_accuracy * 100:.2f}% from fold {best_fold.fold_index})."
        )
        promotion_message_emitted = True

    final_stage = run_full_dataset_training(best_fold)
    if final_stage is not None:
        final_run_tag = "__".join([part for part in [args.run_tag, "final_full"] if part]) or None
        final_run_dir = registry.create_run_directory(final_stage["accuracy"], final_run_tag)
        final_model_to_save = build_model(target_device=torch.device("cpu"))
        final_model_to_save.load_state_dict(final_stage["model_state"])
        final_metadata = final_stage["metadata"]
        final_metrics = final_stage["metrics"]
        final_metadata["run_tag"] = final_run_tag if final_run_tag is not None else args.run_tag
        final_metadata["sample_outputs"] = final_stage["sample_outputs"]
        final_metrics["run_tag"] = final_run_tag if final_run_tag is not None else args.run_tag
        save_run_artifacts(
            final_model_to_save,
            final_metadata,
            final_metrics,
            final_run_dir,
            model_name=args.model_name,
            tolerance=args.promotion_tolerance,
        )
        if registry.is_improvement(final_stage["accuracy"]):
            registry.promote(final_run_dir)
            print(
                f"Promoted consolidated full-data model to {registry.best_dir} (training accuracy {final_stage['accuracy'] * 100:.2f}%)."
            )
            promotion_message_emitted = True
        else:
            print("Archived full-data consolidation run without promotion.")
    if not promotion_message_emitted:
        if previous_best is not None:
            print(
                "No promotion: cross-validation best accuracy "
                f"{best_fold.val_accuracy * 100:.2f}% did not exceed the existing best {previous_best * 100:.2f}%."
            )
        else:
            print("No previous Orion checkpoint detected; stored cross-validation runs without promotion.")

    if len(fold_results) > 1:
        print(
            f"\nCross-validation summary: mean validation accuracy {cv_mean * 100:.2f}% "
            f"(std {cv_std * 100:.2f}%) across {len(fold_results)} folds."
        )

    print("\nSample responses (best fold model):")
    for entry in best_fold.sample_outputs:
        print(f"Input: {entry['input']}")
        print(f"  Predicted label: {entry['predicted_label']}")
        print(f"  Response: {entry['response']}\n")


if __name__ == "__main__":
    main()
