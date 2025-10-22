"""Train an intent classification model with advanced optimisation features.

This script builds a text classification model using PyTorch. It reads the CSV
dataset in ``data/intent_dataset.csv``, tokenises the text, trains a
bidirectional LSTM enhanced with multi-head self-attention, and saves both the
trained model weights and the vocabulary/label metadata to the ``models``
directory.

Compared to the initial baseline, the trainer now supports modern optimisation
features: AdamW with configurable weight decay, label smoothing, gradient
clipping, optional mixed-precision training, cosine/One-Cycle schedulers, early
stopping, and a self-training loop that can pseudo-label unlabelled examples to
keep improving over time.

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
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler


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

TRAINER_VERSION = "orion-trainer-0.3"

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
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'transformers' package is required for the transformer encoder. "
            "Install it via 'pip install transformers'."
        ) from exc
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


@dataclass
class EncodedExample:
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    label: int
    weight: float


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
        embedding_model: Optional[Callable[[str], Sequence[float]]] = None,
    ) -> None:
        self.examples: List[EncodedExample] = []
        if sample_weights is None:
            sample_weights = [1.0] * len(texts)
        if len(sample_weights) != len(texts):
            raise ValueError("Sample weights must match the number of texts.")
        for text, label, weight in zip(texts, labels, sample_weights):
            if embedding_model is not None:
                vector = embedding_model(text)
                token_tensor = torch.tensor(vector, dtype=torch.float32)
                mask_tensor = torch.ones_like(token_tensor)
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
            self.examples.append(
                EncodedExample(
                    tokens=token_tensor,
                    attention_mask=mask_tensor,
                    label=label_id,
                    weight=float(weight),
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        return (
            example.tokens,
            torch.tensor(example.label, dtype=torch.long),
            torch.tensor(example.weight, dtype=torch.float32),
            example.attention_mask,
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
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    amp_enabled = scaler is not None and getattr(scaler, "is_enabled", lambda: False)()

    for inputs, targets, weights, attention_mask in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        weights = weights.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        context = autocast_context(amp_enabled)

        with context:
            logits = model(inputs, attention_mask=attention_mask)
            loss_values = criterion(logits, targets)
            if loss_values.dim() == 0:
                loss_values = loss_values.unsqueeze(0)
            weighted_loss = (loss_values * weights).sum() / weights.sum()

        if amp_enabled and scaler is not None:
            scaler.scale(weighted_loss).backward()
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            weighted_loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()

        batch_loss = loss_values.detach().mean().item()
        total_loss += batch_loss * targets.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, _weights, attention_mask in dataloader:
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
    embedding_model: Optional[Callable[[str], Sequence[float]]] = None,
) -> str:
    if embedding_model is not None:
        vector = embedding_model(text)
        ids = torch.tensor(vector, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.ones_like(ids)
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
        description="Train an intent classification model with advanced optimisation and self-learning capabilities."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/intent_dataset.csv"),
                        help="Path to the labelled CSV dataset containing 'text' and 'label' columns.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"),
                        help="Directory where model artefacts will be recorded.")
    parser.add_argument("--model-name", type=str, default="orion_v0.1",
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
    args = parser.parse_args()

    if not 0.0 < args.self_train_threshold <= 1.0:
        parser.error("--self-train-threshold must be in the interval (0, 1].")
    if args.self_train_weight <= 0:
        parser.error("--self-train-weight must be positive.")
    if args.max_seq_len <= 0:
        parser.error("--max-seq-len must be positive.")
    if args.transformer_learning_rate <= 0:
        parser.error("--transformer-learning-rate must be positive.")

    set_seed(args.seed)

    if not args.dataset.exists():
        raise FileNotFoundError(f"Labelled dataset not found: {args.dataset}")

    dataset_checksum = compute_sha1(args.dataset)

    texts, labels = read_dataset(args.dataset)
    if not texts:
        raise RuntimeError(f"Dataset at {args.dataset} is empty.")

    tokenizer_obj = None
    embedding_fn: Optional[Callable[[str], Sequence[float]]] = None
    sentence_model = None
    sentence_embedding_dim: Optional[int] = None
    embedding_cache_info: Optional[Callable[[], object]] = None
    if args.encoder_type == "transformer":
        tokenizer_obj = load_transformer_tokenizer(args.transformer_model)
        max_seq_len = args.max_seq_len
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

    indices = list(range(len(texts)))
    train_indices, val_indices = stratified_split(indices, labels,
                                                  test_ratio=args.test_ratio,
                                                  seed=args.seed)

    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_weights = [1.0] * len(train_texts)
    base_train_size = len(train_texts)
    supervised_distribution = Counter(train_labels)

    val_texts = [texts[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    if not val_texts:
        raise RuntimeError("Validation split produced no examples; increase --test-ratio or dataset size.")

    val_dataset = IntentDataset(
        val_texts,
        val_labels,
        vocab=vocab,
        label_to_idx=label_to_idx,
        max_len=max_seq_len,
        tokenizer=tokenizer_obj,
        embedding_model=embedding_fn,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.encoder_type == "transformer":
        model = TransformerIntentModel(args.transformer_model, num_classes)
        if tokenizer_obj is not None and hasattr(model, "model") and hasattr(model.model, "resize_token_embeddings"):
            model.model.resize_token_embeddings(len(tokenizer_obj))
        model = model.to(device)
        effective_lr = args.transformer_learning_rate
    elif args.encoder_type == "st":
        if sentence_embedding_dim is None:
            raise RuntimeError("Sentence-transformer embedding dimension could not be determined.")
        model = SentenceTransformerClassifier(
            embedding_dim=sentence_embedding_dim,
            hidden_dim=args.st_hidden_dim,
            num_classes=num_classes,
            dropout=args.st_dropout,
        ).to(device)
        effective_lr = args.learning_rate
    else:
        model = IntentClassifier(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            num_layers=args.encoder_layers,
            attention_heads=args.attention_heads,
            ffn_dim=args.ffn_dim,
        ).to(device)
        effective_lr = args.learning_rate

    criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=args.weight_decay,
    )

    amp_available = GradScaler is not None
    use_amp = bool(args.fp16 and torch.cuda.is_available() and amp_available)
    if args.fp16 and not torch.cuda.is_available():
        print("fp16 requested but CUDA is not available; training with full precision.")
    elif args.fp16 and not amp_available:
        print("fp16 requested but AMP utilities are unavailable; training with full precision.")
    scaler = create_grad_scaler(use_amp)

    unlabeled_texts: List[str] = []
    initial_unlabeled = 0
    unlabeled_checksum: Optional[str] = None
    if args.unlabeled_dataset:
        if not args.unlabeled_dataset.exists():
            raise FileNotFoundError(f"Unlabeled dataset not found: {args.unlabeled_dataset}")
        unlabeled_texts = read_unlabeled_dataset(args.unlabeled_dataset)
        initial_unlabeled = len(unlabeled_texts)
        unlabeled_checksum = compute_sha1(args.unlabeled_dataset)
        print(f"Loaded {initial_unlabeled} unlabeled examples for self-training.")
    elif args.self_train_rounds > 0:
        print("No unlabeled dataset supplied; skipping self-training despite configured rounds.")

    print(
        f"Training on {len(train_texts)} labelled examples across {num_classes} intents; "
        f"validation set has {len(val_texts)} examples."
    )
    if args.encoder_type == "transformer":
        tokenizer_size = len(tokenizer_obj) if tokenizer_obj is not None else "unknown"
        print(
            f"Transformer tokenizer '{args.transformer_model}' with vocabulary size {tokenizer_size} "
            f"and max sequence length {max_seq_len} tokens."
        )
    elif args.encoder_type == "st":
        cache_max = None
        if callable(embedding_cache_info):
            info = embedding_cache_info()
            cache_max = info.maxsize
        cache_desc = cache_max if cache_max not in (None, 0) else "unbounded"
        print(
            f"Sentence-transformer '{args.sentence_transformer_model}' embeddings (dimension {sentence_embedding_dim}, "
            f"cache size {cache_desc})."
        )
    else:
        print(
            f"Vocabulary size: {len(vocab)} (min frequency = {args.min_freq}). "
            f"Max sequence length: {max_seq_len} tokens."
        )

    history: List[Dict[str, object]] = []
    pseudo_rounds: List[Dict[str, object]] = []
    global_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_acc = -float("inf")
    epochs_since_improvement = 0

    def run_stage(stage_name: str, epochs: int) -> bool:
        nonlocal global_epoch, best_state, best_val_acc, epochs_since_improvement
        if epochs <= 0:
            return False
        train_dataset = IntentDataset(
            train_texts,
            train_labels,
            vocab=vocab,
            label_to_idx=label_to_idx,
            max_len=max_seq_len,
            sample_weights=train_weights,
            tokenizer=tokenizer_obj,
            embedding_model=embedding_fn,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        scheduler, per_batch = create_scheduler(
            optimizer,
            args.scheduler,
            epochs,
            len(train_loader),
            effective_lr,
        )
        for local_epoch in range(1, epochs + 1):
            global_epoch += 1
            train_loss, train_acc = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scheduler=scheduler if per_batch else None,
                scheduler_step_per_batch=per_batch,
                scaler=scaler,
                max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            if scheduler is not None and not per_batch:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {global_epoch:02d} [{stage_name}]: "
                f"train_loss={train_loss:.4f} train_acc={train_acc*100:.1f}% "
                f"val_loss={val_loss:.4f} val_acc={val_acc*100:.1f}% "
                f"lr={current_lr:.2e}"
            )
            history.append({
                "epoch": float(global_epoch),
                "stage": stage_name,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "learning_rate": float(current_lr),
                "train_examples": float(len(train_dataset)),
            })
            if val_acc > best_val_acc + 1e-6:
                best_val_acc = val_acc
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if args.early_stop_patience and epochs_since_improvement >= args.early_stop_patience:
                    print("Early stopping triggered due to stagnant validation performance.")
                    return True
        return False

    stop_training = run_stage("supervised", args.epochs)

    total_pseudo_added = 0
    if not stop_training and unlabeled_texts and args.self_train_rounds > 0 and args.self_train_epochs > 0:
        for round_idx in range(1, args.self_train_rounds + 1):
            confident, unlabeled_texts = pseudo_label_unlabeled(
                model,
                unlabeled_texts,
                vocab=vocab,
                label_to_idx=label_to_idx,
                max_len=max_seq_len,
                device=device,
                threshold=args.self_train_threshold,
                tokenizer=tokenizer_obj,
                embedding_model=embedding_fn,
            )
            if not confident:
                print(f"Self-training round {round_idx}: no predictions met the confidence threshold {args.self_train_threshold}.")
                break
            avg_conf = sum(score for _, _, score in confident) / len(confident)
            print(
                f"Self-training round {round_idx}: added {len(confident)} pseudo-labelled examples "
                f"(avg confidence {avg_conf:.3f}). Remaining unlabeled: {len(unlabeled_texts)}"
            )
            for text, label, _score in confident:
                train_texts.append(text)
                train_labels.append(label)
                train_weights.append(float(args.self_train_weight))
            pseudo_rounds.append({
                "round": float(round_idx),
                "added_examples": float(len(confident)),
                "average_confidence": float(avg_conf),
            })
            total_pseudo_added += len(confident)
            epochs_since_improvement = 0  # reset patience because dataset changed.
            stop_training = run_stage(f"self-train-{round_idx}", args.self_train_epochs)
            if stop_training:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

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
        "test_ratio": args.test_ratio,
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
        "run_tag": args.run_tag,
        "class_distribution_total": dict(sorted(Counter(labels).items())),
        "class_distribution_supervised": dict(sorted(supervised_distribution.items())),
        "class_distribution_final": dict(sorted(Counter(train_labels).items())),
        "validation_distribution": dict(sorted(Counter(val_labels).items())),
        "self_training": {
            "rounds_configured": args.self_train_rounds,
            "rounds_ran": len(pseudo_rounds),
            "epochs_per_round": args.self_train_epochs,
            "confidence_threshold": args.self_train_threshold,
            "pseudo_example_weight": args.self_train_weight,
            "examples_added": total_pseudo_added,
            "round_details": pseudo_rounds,
            "initial_unlabeled": initial_unlabeled,
            "remaining_unlabeled": len(unlabeled_texts),
        },
        "training_history": history,
        "best_val_accuracy": best_val_acc,
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

    if not math.isfinite(best_val_acc):
        raise RuntimeError("Validation accuracy is not finite; unable to persist results.")

    best_entry = max(history, key=lambda row: row["val_accuracy"])
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
        "run_tag": args.run_tag,
        "encoder_type": args.encoder_type,
        "effective_learning_rate": effective_lr,
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

    registry = ModelRegistry(args.model_dir, args.model_name, tolerance=args.promotion_tolerance)
    previous_best = registry.best_accuracy()
    promote = registry.is_improvement(best_val_acc)
    metrics["previous_best_accuracy"] = previous_best
    metrics["promoted_to_orion"] = bool(promote)

    run_dir = registry.create_run_directory(best_val_acc, args.run_tag)
    save_run_artifacts(
        model,
        metadata,
        metrics,
        run_dir,
        model_name=args.model_name,
        tolerance=args.promotion_tolerance,
    )

    if promote:
        registry.promote(run_dir)
        print(f"Promoted run to {registry.best_dir} (validation accuracy {best_val_acc * 100:.2f}%).")
    else:
        if previous_best is not None:
            print(
                "Run retained under models/runs/ because validation accuracy "
                f"{best_val_acc * 100:.2f}% did not exceed the existing best {previous_best * 100:.2f}%."
            )
        else:
            print("No previous Orion checkpoint detected; stored run without promotion.")

    sample_inputs = [
        "Hello there!",
        "Please reboot the system after tonight's update.",
        "What time does the meeting start tomorrow?",
        "Remember to submit your report by Friday.",
        "I love how polished the presentation looked!",
    ]
    print("\nSample responses:")
    for sample in sample_inputs:
        predicted_label = predict_label(
            model,
            sample,
            vocab=vocab,
            label_to_idx=label_to_idx,
            max_len=max_seq_len,
            device=device,
            tokenizer=tokenizer_obj,
            embedding_model=embedding_fn,
        )
        reply = generate_response(predicted_label, sample)
        print(f"Input: {sample}")
        print(f"  Predicted label: {predicted_label}")
        print(f"  Response: {reply}\n")


if __name__ == "__main__":
    main()
