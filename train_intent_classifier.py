"""Train an intent classification model with advanced optimisation features.

This script builds a text classification model using PyTorch. It reads the CSV
dataset in ``data/intent_dataset.csv``, tokenises the text, and trains an
encoder (BiLSTM, transformer, or sentence-transformer head) before saving the
weights and metadata to the ``models`` directory.

Compared to the initial baseline, the trainer now supports modern optimisation
features: AdamW with configurable weight decay, label smoothing, gradient
clipping, optional mixed-precision training, cosine/One-Cycle schedulers, early
stopping, and a self-training loop that can pseudo-label unlabelled examples to
keep improving over time. The trainer also runs synthetic self-play rounds that
build label-aware n-gram generators, draft fresh practice prompts, and vet them
via Monte Carlo self-evaluation so the model earns additional supervision
without external labels. A stratified cross-validation harness exercises every
labelled example, and the pseudo-labelling routine now decays its confidence
threshold while weighting samples by their confidence so that the model keeps
learning in a controlled, self-improving fashion.

To cultivate genuine reasoning skills, the training loop now includes a
meta-cognitive introspection engine. It continuously constructs class-specific
representation prototypes, measures the entropy and margin of every decision,
and injects attraction/repulsion forces that push the encoder to carve out
distinct, emotionally grounded intent manifolds. The introspector also tracks
where the model remains uncertain, curating a curiosity ledger that is rolled
into later optimisation stages so the network keeps stretching into the unknown.

New in this release is an adaptive curriculum engine that monitors per-example
confidence, highlights difficult samples, and progressively increases their
loss weights while tempering easy items. This curriculum applies across
supervised, distillation, and consolidated training phases, emitting detailed
metadata so future runs can audit how the dataset evolved.

To push the model into even more ambitious territories, the trainer also
features a transcendent cognition architect. It maintains long-horizon feature
banks, imagination traces, and transition priors so that each batch is scored
against stability, divergence, foresight, synthesis, and affective coherence.
These additional gradients encourage the encoder to reason about counterfactual
paths, emotional undertones, and multi-step dynamics that extend beyond the
current example, dramatically increasing the compute required but equipping the
system with richer self-directed discovery signals.

To escape the training distribution entirely, a frontier intelligence catalyst
now orchestrates novelty scouting, abstraction bridges, and curiosity-driven
transfer objectives. It continuously blends high-entropy counterexamples,
latent bridge prototypes, and emotion-aligned expectations so the encoder keeps
inventing concepts that never explicitly appeared in the dataset while still
remaining grounded in affective context.

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
import fnmatch
import hashlib
import math
import os
import shutil
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

try:
    _THIS_FILE = Path(__file__).resolve()
except NameError:  # pragma: no cover - __file__ is undefined inside interactive shells
    _THIS_FILE = None


def _script_directory() -> Path:
    """Best-effort location of this script when executed via notebooks/cells."""

    if _THIS_FILE is not None:
        return _THIS_FILE.parent
    return Path.cwd().resolve()

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

TRAINER_VERSION = "orion-trainer-0.7"

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


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


def _is_colab_environment() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ or Path("/content").exists()


def _deduplicate_paths(paths: Iterable[Path]) -> List[Path]:
    unique: List[Path] = []
    seen: Set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _candidate_data_directories() -> List[Tuple[Path, bool]]:
    script_dir = _script_directory()
    repo_data_dir = script_dir / "data"
    cwd = Path.cwd()
    candidates: List[Tuple[Path, bool]] = [
        (cwd, False),
        (script_dir, False),
        (repo_data_dir, False),
    ]

    if _is_colab_environment():
        colab_root = Path("/content")
        candidates.append((colab_root, False))
        colab_files = colab_root / "files"
        candidates.append((colab_files, True))
        sample_data = colab_root / "sample_data"
        candidates.append((sample_data, False))
        drive_root = colab_root / "drive"
        candidates.append((drive_root, False))
        my_drive = drive_root / "MyDrive"
        candidates.append((my_drive, False))

    merged: Dict[str, Tuple[Path, bool]] = {}
    for path, allow_deep in candidates:
        try:
            if not path.exists():
                continue
        except OSError:
            continue
        key = str(path)
        if key in merged:
            existing_path, existing_flag = merged[key]
            merged[key] = (existing_path, existing_flag or allow_deep)
        else:
            merged[key] = (path, allow_deep)
    return list(merged.values())


def _strip_ipykernel_arguments(argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Remove the automatic connection arguments that Jupyter/Colab inject."""

    filtered: List[str] = []
    ignored: List[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token in {"-f", "--f"}:
            next_value = argv[idx + 1] if idx + 1 < len(argv) else None
            if next_value is not None and next_value.endswith(".json"):
                ignored.extend([token, next_value])
                idx += 2
                continue
        elif token.startswith(("-f=", "--f=")):
            value = token.split("=", 1)[1]
            if value.endswith(".json"):
                ignored.append(token)
                idx += 1
                continue

        filtered.append(token)
        idx += 1

    return filtered, ignored


def _search_in_directory(
    base: Path,
    names: Sequence[str],
    patterns: Sequence[str],
    *,
    allow_deep: bool,
) -> Optional[Path]:
    sanitized_names = [name for name in names if name]
    lower_names = {name.lower() for name in sanitized_names}

    for name in sanitized_names:
        candidate = base / name
        if candidate.exists():
            return candidate

    try:
        entries = list(base.iterdir())
    except OSError:
        entries = []

    if lower_names:
        for entry in entries:
            try:
                if entry.is_file() and entry.name.lower() in lower_names:
                    return entry
            except OSError:
                continue

    if not allow_deep:
        return None

    csv_entries: List[Path] = []
    for entry in entries:
        try:
            if entry.is_file():
                csv_entries.append(entry)
        except OSError:
            continue

    pattern_matches: List[Path] = []
    for entry in csv_entries:
        if any(fnmatch.fnmatch(entry.name, pattern) for pattern in patterns if pattern):
            pattern_matches.append(entry)

    if pattern_matches:
        pattern_matches.sort(key=lambda p: p.name)
        return pattern_matches[0]

    if len(csv_entries) == 1:
        return csv_entries[0]

    for entry in entries:
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue

        for name in sanitized_names:
            candidate = entry / name
            if candidate.exists():
                return candidate

        try:
            sub_entries = list(entry.iterdir())
        except OSError:
            continue

        for sub_entry in sub_entries:
            try:
                if sub_entry.is_file():
                    if lower_names and sub_entry.name.lower() in lower_names:
                        return sub_entry
                    if any(fnmatch.fnmatch(sub_entry.name, pattern) for pattern in patterns if pattern):
                        return sub_entry
            except OSError:
                continue

    return None


def resolve_training_input_path(
    path: Path,
    *,
    description: str,
    flag: Optional[str] = None,
    search_names: Optional[Sequence[str]] = None,
) -> Path:
    expanded = path.expanduser()
    original_target = expanded.resolve(strict=False)

    script_dir = _script_directory()
    candidates = _deduplicate_paths([
        expanded,
        script_dir / expanded,
    ])

    checked: List[Path] = []
    for candidate in candidates:
        checked.append(candidate)
        try:
            if candidate.exists():
                resolved = candidate.resolve()
                if resolved != original_target:
                    print(f"Resolved {description} path to {resolved}")
                return resolved
        except OSError:
            continue

    names = list(dict.fromkeys([*(search_names or []), expanded.name]))
    patterns: List[str] = []
    suffix = expanded.suffix.lstrip(".")
    if suffix:
        patterns.append(f"{expanded.stem}*.{suffix}")

    for base, allow_deep in _candidate_data_directories():
        found = _search_in_directory(base, names, patterns, allow_deep=allow_deep)
        if found:
            resolved = found.resolve()
            if resolved != original_target:
                location_hint = f" (auto-detected in {found.parent})"
                print(f"Resolved {description} path to {resolved}{location_hint}")
            return resolved
        checked.append(base / expanded.name)

    search_locations = [str(candidate) for candidate in _deduplicate_paths(checked)]
    message_lines = [f"Could not locate {description} file '{expanded.name}'."]
    if search_locations:
        message_lines.append("Looked in:")
        for loc in search_locations[:10]:
            message_lines.append(f"  - {loc}")
        if len(search_locations) > 10:
            message_lines.append("  - ...")
    if flag:
        message_lines.append(f"Provide the path explicitly via {flag}.")
    raise FileNotFoundError("\n".join(message_lines))


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


@dataclass
class LabelNgramModel:
    label: str
    order: int
    transitions: Dict[Tuple[str, ...], Tuple[List[str], List[float]]]
    start_states: Tuple[List[Tuple[str, ...]], List[float]]


@dataclass
class SelfPlayCandidateEvaluation:
    label: str
    blended_confidence: float
    deterministic_confidence: float
    mc_confidence: float
    consistency: float
    margin: float
    top_predictions: List[Tuple[str, float]]
    average_distribution: Dict[str, float]


def build_label_ngram_models(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    order: int = 3,
) -> Dict[str, LabelNgramModel]:
    order = max(1, order)
    sequences: Dict[str, List[List[str]]] = defaultdict(list)
    for text, label in zip(texts, labels):
        tokens = tokenize(text)
        if tokens:
            sequences[label].append(tokens)

    models: Dict[str, LabelNgramModel] = {}
    for label, label_sequences in sequences.items():
        transitions: Dict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
        start_counter: Counter[Tuple[str, ...]] = Counter()
        for tokens in label_sequences:
            if order > 1:
                padded = [BOS_TOKEN] * (order - 1) + tokens + [EOS_TOKEN]
            else:
                padded = tokens + [EOS_TOKEN]
            start_state = tuple(padded[: max(order - 1, 0)])
            start_counter[start_state] += 1
            limit = len(padded) - order + 1
            for idx in range(max(1, limit)):
                prefix = tuple(padded[idx : idx + order - 1]) if order > 1 else tuple()
                next_token = padded[idx + order - 1] if order > 0 else padded[idx]
                transitions[prefix][next_token] += 1
        if not transitions:
            continue
        prepared: Dict[Tuple[str, ...], Tuple[List[str], List[float]]] = {}
        for prefix, counter in transitions.items():
            tokens_list = list(counter.keys())
            weights = [float(counter[token]) for token in tokens_list]
            prepared[prefix] = (tokens_list, weights)
        start_population = list(start_counter.keys())
        start_weights = [float(start_counter[state]) for state in start_population]
        models[label] = LabelNgramModel(
            label=label,
            order=order,
            transitions=prepared,
            start_states=(start_population, start_weights),
        )
    return models


def sample_synthetic_tokens(
    model: LabelNgramModel,
    rng: random.Random,
    *,
    max_tokens: int,
    temperature: float,
) -> List[str]:
    if not model.transitions:
        return []
    safe_max = max(1, max_tokens)
    order = model.order
    starts, weights = model.start_states
    if starts:
        prefix = tuple(rng.choices(starts, weights=weights, k=1)[0])
    else:
        prefix = tuple([BOS_TOKEN] * (order - 1)) if order > 1 else tuple()
    generated: List[str] = []
    inv_temp: Optional[float]
    if temperature not in (0, 1.0):
        inv_temp = 1.0 / max(temperature, 1e-6)
    else:
        inv_temp = None
    max_steps = safe_max * 4
    for _ in range(max_steps):
        if len(generated) >= safe_max:
            break
        key = prefix if order > 1 else tuple()
        if key not in model.transitions:
            break
        options, base_weights = model.transitions[key]
        if not options:
            break
        if inv_temp is not None:
            adjusted = [max(weight, 1e-8) ** inv_temp for weight in base_weights]
        else:
            adjusted = base_weights
        next_token = rng.choices(options, weights=adjusted, k=1)[0]
        if next_token == EOS_TOKEN:
            break
        if next_token not in (BOS_TOKEN, EOS_TOKEN):
            generated.append(next_token)
        if order > 1:
            prefix = (*prefix[1:], next_token)
        if len(generated) >= safe_max:
            break
    return generated


def render_synthetic_text(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    contractions = {
        " n't": "n't",
        " 're": "'re",
        " 's": "'s",
        " 've": "'ve",
        " 'd": "'d",
        " 'll": "'ll",
        " 'm": "'m",
    }
    for old, new in contractions.items():
        text = text.replace(old, new)
    text = text.replace(" ,", ",")
    text = text.replace(" ;", ";")
    text = text.replace(" :", ":")
    text = text.replace(" '", "'")
    text = re.sub(r"\bi\b", "I", text)
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text[-1] not in ".?!":
        text += "."
    return text


def build_label_concept_library(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    label_to_idx: Dict[str, int],
    max_keywords: int = 16,
) -> Tuple[List[Dict[str, float]], List[List[str]]]:
    """Construct per-label lexical concept profiles using TF-IDF weights."""

    num_classes = len(label_to_idx)
    lexical_profiles: List[Dict[str, float]] = [dict() for _ in range(num_classes)]
    keywords: List[List[str]] = [[] for _ in range(num_classes)]
    token_counts: Dict[str, Counter[str]] = {label: Counter() for label in label_to_idx}
    document_frequency: Counter[str] = Counter()

    for text, label in zip(texts, labels):
        if label not in label_to_idx:
            continue
        tokens = tokenize(text)
        if not tokens:
            continue
        token_counts[label].update(tokens)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            document_frequency[token] += 1
    total_documents = max(1, len(texts))
    for label, counter in token_counts.items():
        idx = label_to_idx[label]
        if not counter:
            continue
        total = sum(counter.values())
        if total <= 0:
            continue
        weighted: Dict[str, float] = {}
        for token, freq in counter.items():
            tf = freq / total
            idf = math.log(1.0 + total_documents / (1.0 + document_frequency[token]))
            weighted[token] = tf * idf
        lexical_profiles[idx] = weighted
        if weighted:
            top_tokens = sorted(weighted.items(), key=lambda item: item[1], reverse=True)[:max_keywords]
            keywords[idx] = [token for token, _ in top_tokens]

    return lexical_profiles, keywords


def evaluate_self_play_candidate(
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
    samples: int = 4,
    emotion_encoder: Optional[EmotionLexicon] = None,
    emotion_dim: int = 0,
    emotion_config: Optional[EmotionTrainingConfig] = None,
) -> Optional[SelfPlayCandidateEvaluation]:
    shots = max(1, samples)
    ids, mask, emotion_features = _prepare_model_inputs(
        text,
        vocab=vocab,
        max_len=max_len,
        device=device,
        tokenizer=tokenizer,
        tokenizer_cache=tokenizer_cache,
        embedding_model=embedding_model,
        emotion_encoder=emotion_encoder,
        emotion_dim=emotion_dim,
    )
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    counts: Counter[str] = Counter()
    confidences: List[float] = []
    margins: List[float] = []
    prob_sums: Dict[str, float] = defaultdict(float)
    was_training = model.training
    model.train()
    try:
        with torch.no_grad():
            for _ in range(shots):
                supports_emotion = (
                    emotion_features is not None
                    and emotion_features.numel() > 0
                    and getattr(model, "supports_emotion_features", False)
                    and (emotion_config is None or emotion_config.enabled)
                )
                if supports_emotion:
                    logits, _, _ = model(
                        ids,
                        attention_mask=mask,
                        emotion_features=emotion_features,
                        return_components=True,
                    )
                else:
                    logits = model(ids, attention_mask=mask)
                probs = torch.softmax(logits, dim=-1)
                confidence_tensor, predicted_idx = probs.max(dim=-1)
                label = idx_to_label[predicted_idx.item()]
                counts[label] += 1
                confidence = float(confidence_tensor.item())
                confidences.append(confidence)
                if probs.shape[-1] >= 2:
                    top_values = probs.topk(min(2, probs.shape[-1]), dim=-1).values[0]
                    if top_values.shape[0] == 2:
                        margin = float(top_values[0].item() - top_values[1].item())
                    else:
                        margin = float(top_values[0].item())
                else:
                    margin = confidence
                margins.append(margin)
                for idx, prob in enumerate(probs[0]):
                    prob_sums[idx_to_label[idx]] += float(prob.item())
    finally:
        model.train(was_training)

    total = sum(counts.values())
    if total == 0:
        return None
    restore_mode = model.training
    prediction = predict_with_trace(
        model,
        text,
        vocab=vocab,
        label_to_idx=label_to_idx,
        max_len=max_len,
        device=device,
        tokenizer=tokenizer,
        tokenizer_cache=tokenizer_cache,
        embedding_model=embedding_model,
        top_k=3,
        emotion_encoder=emotion_encoder,
        emotion_dim=emotion_dim,
        emotion_config=emotion_config,
    )
    model.train(restore_mode)

    matches = counts[prediction.label]
    consistency = matches / total if total else 0.0
    mc_confidence = sum(confidences) / total if confidences else 0.0
    avg_margin = sum(margins) / len(margins) if margins else 0.0
    blended_confidence = (prediction.confidence + mc_confidence) / 2.0
    average_distribution = {
        label: prob_sums[label] / total for label in prob_sums
    }
    return SelfPlayCandidateEvaluation(
        label=prediction.label,
        blended_confidence=blended_confidence,
        deterministic_confidence=prediction.confidence,
        mc_confidence=mc_confidence,
        consistency=consistency,
        margin=avg_margin,
        top_predictions=prediction.top_predictions,
        average_distribution=average_distribution,
    )


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


class EmotionLexicon:
    """Lightweight affective lexicon that maps tokens to multi-emotion weights."""

    DEFAULT_LEXICON: Dict[str, Dict[str, float]] = {
        "happy": {"joy": 1.0, "anticipation": 0.25},
        "happier": {"joy": 1.1, "anticipation": 0.35},
        "happiest": {"joy": 1.15, "anticipation": 0.4},
        "joy": {"joy": 1.0},
        "joyful": {"joy": 1.1, "anticipation": 0.2},
        "delight": {"joy": 0.9, "surprise": 0.2},
        "delighted": {"joy": 1.0, "surprise": 0.25},
        "excited": {"anticipation": 1.0, "joy": 0.6},
        "thrilled": {"anticipation": 1.0, "joy": 0.7},
        "optimistic": {"anticipation": 0.8, "joy": 0.5, "trust": 0.6},
        "hope": {"anticipation": 0.9, "trust": 0.4},
        "hopeful": {"anticipation": 1.0, "trust": 0.5},
        "calm": {"trust": 0.8, "joy": 0.2},
        "relaxed": {"trust": 0.7, "joy": 0.35},
        "grateful": {"joy": 0.6, "trust": 0.9},
        "thanks": {"joy": 0.5, "trust": 0.6},
        "appreciate": {"joy": 0.5, "trust": 0.6},
        "love": {"joy": 1.1, "trust": 0.7},
        "lovely": {"joy": 0.9, "trust": 0.5},
        "awesome": {"joy": 0.9, "surprise": 0.6},
        "amazing": {"joy": 0.9, "surprise": 0.8},
        "wow": {"surprise": 1.0, "joy": 0.35},
        "surprised": {"surprise": 1.0, "anticipation": 0.35},
        "curious": {"anticipation": 0.8, "surprise": 0.4},
        "interested": {"anticipation": 0.6, "trust": 0.4},
        "focused": {"anticipation": 0.5, "trust": 0.6},
        "motivated": {"anticipation": 0.7, "joy": 0.45},
        "energetic": {"joy": 0.6, "anticipation": 0.6},
        "proud": {"joy": 0.7, "trust": 0.5},
        "confident": {"trust": 0.9, "joy": 0.3},
        "secure": {"trust": 0.8},
        "support": {"trust": 0.7, "joy": 0.35},
        "supportive": {"trust": 0.9, "joy": 0.3},
        "reassure": {"trust": 0.7, "joy": 0.25},
        "care": {"joy": 0.45, "trust": 0.55},
        "caring": {"joy": 0.5, "trust": 0.65},
        "calming": {"trust": 0.7, "joy": 0.2},
        "sad": {"sadness": 1.0},
        "depressed": {"sadness": 1.1, "fear": 0.4},
        "unhappy": {"sadness": 0.9},
        "cry": {"sadness": 0.9, "fear": 0.3},
        "crying": {"sadness": 1.0, "fear": 0.3},
        "mourn": {"sadness": 1.0},
        "lonely": {"sadness": 0.9, "fear": 0.4},
        "broken": {"sadness": 0.8, "anger": 0.4},
        "tired": {"sadness": 0.55, "disgust": 0.2},
        "exhausted": {"sadness": 0.6, "disgust": 0.3},
        "bored": {"sadness": 0.55},
        "disappointed": {"sadness": 0.7, "anger": 0.4},
        "frustrated": {"anger": 0.9, "sadness": 0.4},
        "angry": {"anger": 1.0},
        "mad": {"anger": 0.95},
        "furious": {"anger": 1.15, "disgust": 0.4},
        "annoyed": {"anger": 0.75},
        "irritated": {"anger": 0.7},
        "upset": {"sadness": 0.8, "anger": 0.45},
        "hate": {"anger": 0.9, "disgust": 0.6},
        "hated": {"anger": 0.85, "disgust": 0.55},
        "scared": {"fear": 1.0},
        "afraid": {"fear": 1.0},
        "terrified": {"fear": 1.2},
        "worried": {"fear": 0.8, "anticipation": 0.35},
        "nervous": {"fear": 0.85, "anticipation": 0.25},
        "anxious": {"fear": 0.9, "anticipation": 0.35},
        "panic": {"fear": 1.1},
        "uncertain": {"fear": 0.5, "anticipation": 0.4},
        "confused": {"surprise": 0.5, "fear": 0.4},
        "shocked": {"surprise": 1.0, "fear": 0.55},
        "stunned": {"surprise": 0.95},
        "amazed": {"surprise": 0.9, "joy": 0.6},
        "astonished": {"surprise": 1.0},
        "intrigued": {"anticipation": 0.6, "surprise": 0.4},
        "disgusted": {"disgust": 1.0},
        "gross": {"disgust": 0.9},
        "nasty": {"disgust": 0.95},
        "dirty": {"disgust": 0.6},
        "sick": {"disgust": 0.7, "sadness": 0.4},
        "burnout": {"sadness": 0.7, "disgust": 0.4},
        "burned": {"sadness": 0.65, "anger": 0.45},
        "together": {"trust": 0.6, "joy": 0.35},
        "team": {"trust": 0.6, "anticipation": 0.35},
        "collaborate": {"trust": 0.55, "anticipation": 0.45},
        "learn": {"anticipation": 0.65, "joy": 0.3},
        "learning": {"anticipation": 0.7, "joy": 0.35},
        "grow": {"anticipation": 0.6, "joy": 0.35},
        "improve": {"anticipation": 0.7, "trust": 0.4},
        "progress": {"anticipation": 0.7, "joy": 0.4},
        "celebrate": {"joy": 0.9, "anticipation": 0.5},
        "celebrating": {"joy": 1.0, "anticipation": 0.55},
        "success": {"joy": 0.75, "anticipation": 0.6, "trust": 0.5},
        "win": {"joy": 0.9, "anticipation": 0.5},
        "winning": {"joy": 1.0, "anticipation": 0.55},
        "fail": {"sadness": 0.9, "disgust": 0.4},
        "failing": {"sadness": 1.0, "disgust": 0.45},
        "failure": {"sadness": 0.95, "disgust": 0.4},
        "mistake": {"sadness": 0.6, "fear": 0.4},
        "fix": {"anticipation": 0.55, "trust": 0.5},
        "solved": {"joy": 0.7, "trust": 0.4},
        "resolved": {"joy": 0.7, "trust": 0.5},
        "urgent": {"fear": 0.7, "anticipation": 0.6},
        "critical": {"fear": 0.65, "anticipation": 0.55},
        "blocked": {"sadness": 0.6, "anger": 0.5},
        "delay": {"sadness": 0.55, "anticipation": 0.4},
        "waiting": {"anticipation": 0.6, "sadness": 0.35},
        "patience": {"trust": 0.55, "anticipation": 0.3},
        "cheer": {"joy": 0.8, "anticipation": 0.45},
        "cheerful": {"joy": 0.9, "anticipation": 0.35},
        "encourage": {"joy": 0.55, "trust": 0.6},
        "encouraging": {"joy": 0.6, "trust": 0.6},
        "empower": {"anticipation": 0.6, "joy": 0.4, "trust": 0.5},
        "empowered": {"anticipation": 0.65, "joy": 0.45, "trust": 0.5},
        "focus": {"anticipation": 0.6, "trust": 0.5},
        "inspire": {"joy": 0.6, "anticipation": 0.55},
        "inspired": {"joy": 0.7, "anticipation": 0.6},
        "motivating": {"joy": 0.6, "anticipation": 0.7},
        "spark": {"surprise": 0.6, "anticipation": 0.45},
    }

    EMOTIONS: Tuple[str, ...] = (
        "joy",
        "trust",
        "fear",
        "surprise",
        "sadness",
        "disgust",
        "anger",
        "anticipation",
    )

    def __init__(self, lexicon: Optional[Dict[str, Dict[str, float]]] = None, *, smoothing: float = 1e-3) -> None:
        self.lexicon = lexicon or self.DEFAULT_LEXICON
        self.emotions: Tuple[str, ...] = self.EMOTIONS
        self.emotion_to_idx: Dict[str, int] = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        self.smoothing = float(max(smoothing, 1e-6))

    def vectorise(self, text: str) -> List[float]:
        tokens = tokenize(text)
        scores = [0.0 for _ in self.emotions]
        if not tokens:
            return scores
        emphasis = self._compute_emphasis(text)
        for token in tokens:
            info = self.lexicon.get(token)
            if not info:
                continue
            for emotion, value in info.items():
                idx = self.emotion_to_idx.get(emotion)
                if idx is None:
                    continue
                scores[idx] += float(value)
        for emotion, delta in emphasis.items():
            idx = self.emotion_to_idx.get(emotion)
            if idx is not None:
                scores[idx] += delta
        norm = sum(scores)
        if norm <= 0:
            return scores
        denom = norm + self.smoothing * len(scores)
        return [value / denom for value in scores]

    def _compute_emphasis(self, text: str) -> Dict[str, float]:
        lowered = normalise_text(text)
        emphasis: Dict[str, float] = {}
        exclamations = lowered.count("!")
        questions = lowered.count("?")
        uppercase_chars = sum(1 for ch in text if ch.isalpha() and ch == ch.upper())
        alpha_chars = sum(1 for ch in text if ch.isalpha())
        uppercase_ratio = (uppercase_chars / alpha_chars) if alpha_chars else 0.0

        if exclamations:
            emphasis["joy"] = emphasis.get("joy", 0.0) + 0.25 * exclamations
            emphasis["anticipation"] = emphasis.get("anticipation", 0.0) + 0.15 * exclamations
        if questions:
            emphasis["anticipation"] = emphasis.get("anticipation", 0.0) + 0.1 * questions
        if uppercase_ratio > 0.55:
            emphasis["anger"] = emphasis.get("anger", 0.0) + uppercase_ratio * 0.6
        if any(face in text for face in (":)", "😊", "😀", "😁", "❤️")):
            emphasis["joy"] = emphasis.get("joy", 0.0) + 0.6
            emphasis["trust"] = emphasis.get("trust", 0.0) + 0.2
        if any(face in text for face in (":(", "😢", "😭", "💔")):
            emphasis["sadness"] = emphasis.get("sadness", 0.0) + 0.7
        if any(face in text for face in (">:(", "😡", "🤬")):
            emphasis["anger"] = emphasis.get("anger", 0.0) + 0.8
        if any(face in text for face in ("😱", "😨", "😰")):
            emphasis["fear"] = emphasis.get("fear", 0.0) + 0.65
        return emphasis


class EmotionPrototypeMemory:
    """Track emotion prototypes per intent to enable affect-guided reasoning."""

    def __init__(self, num_classes: int, num_emotions: int, *, smoothing: float = 1e-3) -> None:
        self.num_classes = int(num_classes)
        self.num_emotions = int(num_emotions)
        self.smoothing = float(max(smoothing, 1e-6))
        self.prototype_sums = torch.zeros(num_classes, num_emotions, dtype=torch.float32)
        self.prototype_weights = torch.zeros(num_classes, dtype=torch.float32)
        self.total_updates = 0

    def reset(self) -> None:
        self.prototype_sums.zero_()
        self.prototype_weights.zero_()
        self.total_updates = 0

    def register_vector(self, label_idx: int, vector: torch.Tensor, weight: float = 1.0) -> None:
        if vector.numel() != self.num_emotions:
            return
        idx = int(label_idx)
        if not (0 <= idx < self.num_classes):
            return
        w = float(max(weight, 0.0))
        if w == 0.0:
            return
        self.prototype_sums[idx] += vector.detach().cpu() * w
        self.prototype_weights[idx] += w
        self.total_updates += 1

    def register_vectors(
        self,
        label_indices: Sequence[int],
        vectors: Sequence[Sequence[float] | torch.Tensor],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        if weights is None:
            weights = [1.0] * len(label_indices)
        for idx, vector, weight in zip(label_indices, vectors, weights):
            tensor_vec = (
                vector if isinstance(vector, torch.Tensor) else torch.tensor(list(vector), dtype=torch.float32)
            )
            self.register_vector(idx, tensor_vec, float(weight))

    def register_texts(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        label_to_idx: Dict[str, int],
        encoder: EmotionLexicon,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        if weights is None:
            weights = [1.0] * len(texts)
        for text, label, weight in zip(texts, labels, weights):
            idx = label_to_idx.get(label)
            if idx is None:
                continue
            vector = torch.tensor(encoder.vectorise(text), dtype=torch.float32)
            self.register_vector(idx, vector, float(weight))

    def prototypes_tensor(self, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        denom = (self.prototype_weights.unsqueeze(1) + self.smoothing).clamp_min(1e-6)
        prototypes = self.prototype_sums / denom
        if device is not None or dtype is not None:
            prototypes = prototypes.to(device=device or prototypes.device, dtype=dtype or prototypes.dtype)
        return prototypes

    def alignment_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        prototypes = self.prototypes_tensor(device=logits.device, dtype=logits.dtype)
        scaled_logits = logits / max(temperature, 1e-6)
        probs = torch.softmax(scaled_logits, dim=-1)
        expected = probs @ prototypes
        diff = expected - targets
        return (diff * diff).mean(dim=-1)

    def summary(self) -> Dict[str, object]:
        return {
            "total_updates": int(self.total_updates),
            "mean_weight": float(self.prototype_weights.mean().item()) if self.num_classes else 0.0,
            "min_weight": float(self.prototype_weights.min().item()) if self.num_classes else 0.0,
            "max_weight": float(self.prototype_weights.max().item()) if self.num_classes else 0.0,
        }


@dataclass
class EmotionTrainingConfig:
    memory: EmotionPrototypeMemory
    weight: float
    temperature: float
    enabled: bool


@dataclass
class MetaCognitiveConfig:
    introspector: "MetaCognitiveIntrospector"
    enabled: bool
    attraction_weight: float
    repulsion_weight: float
    discovery_weight: float
    gap_margin: float
    temperature: float


@dataclass
class NeuroSymbolicConfig:
    reasoner: "NeuroSymbolicReasoner"
    enabled: bool
    structural_weight: float
    semantic_weight: float
    affective_weight: float
    temperature: float
    self_loop: float


@dataclass
class SelfDiscoveryConfig:
    orchestrator: "SelfDiscoveryOrchestrator"
    enabled: bool
    alignment_weight: float
    contrast_weight: float
    imagination_weight: float
    emotion_weight: float
    temperature: float
    min_confidence: float
    margin: float


@dataclass
class TranscendentCognitionConfig:
    architect: "TranscendentCognitionEngine"
    enabled: bool
    stability_weight: float
    divergence_weight: float
    foresight_weight: float
    synthesis_weight: float
    affective_weight: float
    entropy_weight: float
    temperature: float
    margin: float


@dataclass
class FrontierIntelligenceConfig:
    catalyst: "FrontierIntelligenceEngine"
    enabled: bool
    novelty_weight: float
    abstraction_weight: float
    transfer_weight: float
    curiosity_weight: float
    emotion_weight: float
    meta_weight: float
    temperature: float
    margin: float


class MetaCognitiveIntrospector:
    """Maintain class prototypes and meta-cognitive diagnostics."""

    def __init__(
        self,
        num_classes: int,
        *,
        momentum: float = 0.1,
        margin: float = 1.0,
        history_limit: int = 64,
    ) -> None:
        self.num_classes = int(max(1, num_classes))
        self.momentum = float(max(1e-4, min(momentum, 0.999)))
        self.margin = float(max(1e-6, margin))
        self.history: deque[Dict[str, object]] = deque(maxlen=max(10, history_limit))
        self.feature_dim: Optional[int] = None
        self.device: Optional[torch.device] = None
        self.prototypes: Optional[torch.Tensor] = None
        self.prototype_counts: Optional[torch.Tensor] = None
        self.label_gap = torch.zeros(self.num_classes, dtype=torch.float32)
        self.label_entropy = torch.zeros(self.num_classes, dtype=torch.float32)
        self.avg_gap: float = 0.0
        self.avg_entropy: float = 0.0
        self.total_updates: int = 0

    def ensure_buffers(self, feature_dim: int, device: torch.device) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive for meta-introspection")
        if self.prototypes is None or self.prototypes.shape[1] != feature_dim:
            self.feature_dim = feature_dim
            self.device = device
            self.prototypes = torch.zeros(self.num_classes, feature_dim, device=device)
            self.prototype_counts = torch.zeros(self.num_classes, device=device)
        elif self.prototypes.device != device:
            self.prototypes = self.prototypes.to(device)
            if self.prototype_counts is not None:
                self.prototype_counts = self.prototype_counts.to(device)
            self.device = device

    def coverage(self) -> float:
        if self.prototype_counts is None or self.prototype_counts.numel() == 0:
            return 0.0
        return float(((self.prototype_counts > 0).float().mean()).item())

    def compute_regulariser(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        config: MetaCognitiveConfig,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch = features.shape[0]
        if batch == 0:
            zero = torch.zeros(0, device=features.device, dtype=features.dtype)
            return zero, {
                "loss": 0.0,
                "attraction": 0.0,
                "repulsion": 0.0,
                "novelty": 0.0,
                "gap": 0.0,
                "entropy": 0.0,
                "coverage": self.coverage(),
                "samples": 0,
            }

        self.ensure_buffers(features.shape[1], features.device)
        assert self.prototypes is not None and self.prototype_counts is not None

        prototypes = self.prototypes.detach()
        counts = self.prototype_counts.detach()
        coverage = self.coverage()

        available_same = counts[labels] > 0
        same_proto = prototypes[labels]
        attraction = (features - same_proto).pow(2).sum(dim=1)
        attraction = attraction * available_same.float()

        prototype_mask = counts > 0
        if prototype_mask.any():
            diff = features.unsqueeze(1) - prototypes.unsqueeze(0)
            dist_sq = diff.pow(2).sum(dim=-1)
            same_mask = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).bool()
            effective_mask = prototype_mask.unsqueeze(0) & ~same_mask
            repulsion_candidates = torch.relu(self.margin - dist_sq)
            effective_mask_f = effective_mask.float()
            denom = effective_mask_f.sum(dim=1).clamp(min=1.0)
            repulsion = (repulsion_candidates * effective_mask_f).sum(dim=1) / denom
        else:
            repulsion = torch.zeros_like(attraction)

        probs = torch.softmax(logits, dim=-1)
        if probs.shape[1] >= 2:
            top_values, _ = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
            if top_values.shape[1] == 1:
                gap = top_values[:, 0]
            else:
                gap = top_values[:, 0] - top_values[:, 1]
        else:
            gap = probs[:, 0]
        novelty = torch.relu(config.gap_margin - gap)
        entropy = -(probs.clamp_min(1e-9).log() * probs).sum(dim=1)

        total = (
            config.attraction_weight * attraction
            + config.repulsion_weight * repulsion
            + config.discovery_weight * novelty
        )
        temperature = max(1e-6, float(config.temperature))
        total = total / temperature

        summary = {
            "loss": float(total.detach().mean().item()),
            "attraction": float(attraction.detach().mean().item()),
            "repulsion": float(repulsion.detach().mean().item()),
            "novelty": float(novelty.detach().mean().item()),
            "gap": float(gap.detach().mean().item()),
            "entropy": float(entropy.detach().mean().item()),
            "coverage": coverage,
            "samples": batch,
        }
        return total, summary

    def update_memory(self, features: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor) -> None:
        if features.numel() == 0:
            return
        with torch.no_grad():
            self.ensure_buffers(features.shape[1], features.device)
            assert self.prototypes is not None and self.prototype_counts is not None

            probs = torch.softmax(logits, dim=-1)
            if probs.shape[1] >= 2:
                top_values, _ = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
                if top_values.shape[1] == 1:
                    gap = top_values[:, 0]
                else:
                    gap = top_values[:, 0] - top_values[:, 1]
            else:
                gap = probs[:, 0]
            entropy = -(probs.clamp_min(1e-9).log() * probs).sum(dim=1)
            predicted = probs.argmax(dim=1)

            momentum = self.momentum
            for class_idx in labels.unique(sorted=False):
                idx = int(class_idx.item())
                mask = labels == class_idx
                if not mask.any():
                    continue
                class_mean = features[mask].mean(dim=0)
                if self.prototype_counts[idx] <= 0:
                    self.prototypes[idx] = class_mean
                else:
                    self.prototypes[idx] = (
                        (1 - momentum) * self.prototypes[idx] + momentum * class_mean
                    )
                self.prototype_counts[idx] = (
                    (1 - momentum) * self.prototype_counts[idx] + momentum * mask.sum().float()
                )

            mean_gap = float(gap.mean().item()) if gap.numel() > 0 else 0.0
            mean_entropy = float(entropy.mean().item()) if entropy.numel() > 0 else 0.0
            blend = min(0.5, max(self.momentum, 1e-3))
            if self.total_updates == 0:
                self.avg_gap = mean_gap
                self.avg_entropy = mean_entropy
            else:
                self.avg_gap = (1 - blend) * self.avg_gap + blend * mean_gap
                self.avg_entropy = (1 - blend) * self.avg_entropy + blend * mean_entropy
            self.total_updates += int(features.shape[0])

            novelty_scores = (1.0 - gap).detach()
            top_k = min(3, novelty_scores.numel())
            if top_k > 0:
                top_indices = torch.topk(novelty_scores, k=top_k).indices.tolist()
                for idx in top_indices:
                    self.history.append(
                        {
                            "gap": float(gap[idx].item()),
                            "entropy": float(entropy[idx].item()),
                            "confidence": float(probs[idx, predicted[idx]].item()),
                            "label": int(labels[idx].item()),
                            "predicted": int(predicted[idx].item()),
                        }
                    )

            for class_idx in range(self.num_classes):
                mask = predicted == class_idx
                if mask.any():
                    class_gap = float(gap[mask].mean().item())
                    class_entropy = float(entropy[mask].mean().item())
                    self.label_gap[class_idx] = (
                        (1 - blend) * self.label_gap[class_idx] + blend * class_gap
                    )
                    self.label_entropy[class_idx] = (
                        (1 - blend) * self.label_entropy[class_idx] + blend * class_entropy
                    )

    def snapshot(self) -> Dict[str, object]:
        prototypes_list: Optional[List[List[float]]] = None
        counts_list: Optional[List[float]] = None
        if self.prototypes is not None:
            prototypes_list = self.prototypes.detach().cpu().tolist()
        if self.prototype_counts is not None:
            counts_list = self.prototype_counts.detach().cpu().tolist()
        return {
            "num_classes": self.num_classes,
            "feature_dim": int(self.feature_dim or 0),
            "prototypes": prototypes_list,
            "counts": counts_list,
            "average_gap": float(self.avg_gap),
            "average_entropy": float(self.avg_entropy),
            "total_updates": int(self.total_updates),
            "label_gap": [float(x) for x in self.label_gap.tolist()],
            "label_entropy": [float(x) for x in self.label_entropy.tolist()],
            "coverage": self.coverage(),
            "history": list(self.history),
        }

    def load_snapshot(self, snapshot: Dict[str, object], device: torch.device) -> None:
        prototypes = snapshot.get("prototypes")
        counts = snapshot.get("counts")
        feature_dim = int(snapshot.get("feature_dim", 0) or 0)
        if not prototypes or not counts or feature_dim <= 0:
            return
        proto_tensor = torch.tensor(prototypes, dtype=torch.float32, device=device)
        count_tensor = torch.tensor(counts, dtype=torch.float32, device=device)
        self.ensure_buffers(feature_dim, device)
        assert self.prototypes is not None and self.prototype_counts is not None
        self.prototypes.copy_(proto_tensor)
        self.prototype_counts.copy_(count_tensor)
        self.avg_gap = float(snapshot.get("average_gap", self.avg_gap))
        self.avg_entropy = float(snapshot.get("average_entropy", self.avg_entropy))
        self.total_updates = int(snapshot.get("total_updates", self.total_updates))
        label_gap = snapshot.get("label_gap")
        if isinstance(label_gap, list) and len(label_gap) == self.num_classes:
            self.label_gap = torch.tensor(label_gap, dtype=torch.float32)
        label_entropy = snapshot.get("label_entropy")
        if isinstance(label_entropy, list) and len(label_entropy) == self.num_classes:
            self.label_entropy = torch.tensor(label_entropy, dtype=torch.float32)
        history_items = snapshot.get("history", [])
        self.history.clear()
        if isinstance(history_items, list):
            for item in history_items:
                if isinstance(item, dict):
                    cleaned = {
                        "gap": float(item.get("gap", 0.0)),
                        "entropy": float(item.get("entropy", 0.0)),
                        "confidence": float(item.get("confidence", 0.0)),
                        "label": item.get("label"),
                        "predicted": item.get("predicted"),
                    }
                    self.history.append(cleaned)


class NeuroSymbolicReasoner:
    """Fuse lexical concept graphs with representation dynamics."""

    def __init__(
        self,
        num_classes: int,
        *,
        idx_to_label: Sequence[str],
        lexical_profiles: Sequence[Dict[str, float]],
        lexical_keywords: Sequence[Sequence[str]],
        lexical_weight: float = 0.35,
        graph_momentum: float = 0.2,
        feature_momentum: float = 0.12,
        min_confidence: float = 0.2,
        history_limit: int = 96,
        smoothing: float = 1e-4,
        emotion_dim: int = 0,
    ) -> None:
        self.num_classes = int(max(1, num_classes))
        self.idx_to_label = [str(label) for label in idx_to_label]
        self.lexical_keywords = [list(seq) for seq in lexical_keywords]
        self.lexical_weight = float(min(max(lexical_weight, 0.0), 1.0))
        self.graph_momentum = float(min(max(graph_momentum, 0.0), 1.0))
        self.feature_momentum = float(min(max(feature_momentum, 0.0), 1.0))
        self.min_confidence = float(min(max(min_confidence, 0.0), 1.0))
        self.smoothing = float(max(smoothing, 1e-6))
        self.history: deque[Dict[str, object]] = deque(maxlen=max(10, history_limit))
        base_profiles = list(lexical_profiles)
        if len(base_profiles) < self.num_classes:
            base_profiles.extend({} for _ in range(self.num_classes - len(base_profiles)))
        self.lexical_matrix = self._build_lexical_matrix(base_profiles)
        self.feature_dim: Optional[int] = None
        self.emotion_dim = int(max(0, emotion_dim))
        self.device: Optional[torch.device] = None
        self.cooccurrence: Optional[torch.Tensor] = None
        self.graph_counts: Optional[torch.Tensor] = None
        self.concept_prototypes: Optional[torch.Tensor] = None
        self.concept_counts: Optional[torch.Tensor] = None
        self.emotion_prototypes: Optional[torch.Tensor] = None
        self.total_updates: int = 0
        self._pending_snapshot: Optional[Dict[str, object]] = None

    def _build_lexical_matrix(self, profiles: Sequence[Dict[str, float]]) -> torch.Tensor:
        matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float32)
        norms: List[float] = []
        for profile in profiles[: self.num_classes]:
            norm = math.sqrt(sum(value * value for value in profile.values()))
            norms.append(norm)
        while len(norms) < self.num_classes:
            norms.append(0.0)
        for row in range(self.num_classes):
            profile = profiles[row] if row < len(profiles) else {}
            if not profile:
                matrix[row, row] = 1.0
                continue
            for col in range(self.num_classes):
                other = profiles[col] if col < len(profiles) else {}
                if not other:
                    continue
                dot = 0.0
                for token, weight in profile.items():
                    dot += weight * other.get(token, 0.0)
                denom = norms[row] * norms[col]
                if denom > 0 and dot > 0:
                    matrix[row, col] = float(dot / denom)
            if float(matrix[row].sum().item()) <= 0:
                matrix[row, row] = 1.0
        row_sum = matrix.sum(dim=1, keepdim=True)
        safe = torch.where(
            row_sum > 0,
            matrix / row_sum.clamp_min(1e-6),
            torch.full_like(matrix, 1.0 / self.num_classes),
        )
        return safe

    def _apply_pending_snapshot(self) -> None:
        if not self._pending_snapshot or self.device is None:
            return
        snapshot = self._pending_snapshot
        device = self.device
        if self.cooccurrence is not None:
            cooccurrence = snapshot.get("cooccurrence")
            if isinstance(cooccurrence, list):
                tensor = torch.tensor(cooccurrence, dtype=torch.float32, device=device)
                if tensor.shape == self.cooccurrence.shape:
                    self.cooccurrence.copy_(tensor)
        if self.graph_counts is not None:
            graph_counts = snapshot.get("graph_counts")
            if isinstance(graph_counts, list):
                tensor = torch.tensor(graph_counts, dtype=torch.float32, device=device)
                if tensor.shape == self.graph_counts.shape:
                    self.graph_counts.copy_(tensor)
        if self.concept_prototypes is not None:
            concept = snapshot.get("concept_prototypes")
            if isinstance(concept, list):
                tensor = torch.tensor(concept, dtype=torch.float32, device=device)
                if tensor.shape == self.concept_prototypes.shape:
                    self.concept_prototypes.copy_(tensor)
        if self.concept_counts is not None:
            concept_counts = snapshot.get("concept_counts")
            if isinstance(concept_counts, list):
                tensor = torch.tensor(concept_counts, dtype=torch.float32, device=device)
                if tensor.shape == self.concept_counts.shape:
                    self.concept_counts.copy_(tensor)
        emotion_proto = snapshot.get("emotion_prototypes")
        if isinstance(emotion_proto, list):
            tensor = torch.tensor(emotion_proto, dtype=torch.float32, device=device)
            if tensor.dim() == 2:
                self.emotion_dim = tensor.shape[1]
                if self.emotion_prototypes is None or self.emotion_prototypes.shape != tensor.shape:
                    self.emotion_prototypes = torch.zeros_like(tensor)
                self.emotion_prototypes.copy_(tensor)
        history_items = snapshot.get("history", [])
        if isinstance(history_items, list):
            for item in history_items:
                if isinstance(item, dict):
                    cleaned = {
                        "confident": int(item.get("confident", 0)),
                        "mean_confidence": float(item.get("mean_confidence", 0.0)),
                        "graph_entropy": float(item.get("graph_entropy", 0.0)),
                    }
                    self.history.append(cleaned)
        self.total_updates = int(snapshot.get("total_updates", self.total_updates))
        self._pending_snapshot = None

    def ensure_buffers(
        self,
        feature_dim: int,
        device: torch.device,
        emotion_dim: Optional[int] = None,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive for neuro-symbolic reasoning")
        if self.feature_dim is None:
            self.feature_dim = feature_dim
        if self.feature_dim != feature_dim:
            raise ValueError(
                f"Feature dimension changed from {self.feature_dim} to {feature_dim} while neuro-symbolic reasoning is active."
            )
        self.device = device
        if self.cooccurrence is None:
            self.cooccurrence = torch.zeros(self.num_classes, self.num_classes, device=device)
            self.graph_counts = torch.zeros(self.num_classes, device=device)
        else:
            if self.cooccurrence.device != device:
                self.cooccurrence = self.cooccurrence.to(device)
            if self.graph_counts is not None and self.graph_counts.device != device:
                self.graph_counts = self.graph_counts.to(device)
        if self.concept_prototypes is None:
            self.concept_prototypes = torch.zeros(self.num_classes, feature_dim, device=device)
            self.concept_counts = torch.zeros(self.num_classes, device=device)
        else:
            if self.concept_prototypes.shape[1] != feature_dim:
                raise ValueError("Model feature dimension changed during neuro-symbolic reasoning.")
            if self.concept_prototypes.device != device:
                self.concept_prototypes = self.concept_prototypes.to(device)
            if self.concept_counts is not None and self.concept_counts.device != device:
                self.concept_counts = self.concept_counts.to(device)
        resolved_emotion_dim = emotion_dim if emotion_dim is not None else self.emotion_dim
        resolved_emotion_dim = int(max(0, resolved_emotion_dim))
        if resolved_emotion_dim > 0:
            self.emotion_dim = resolved_emotion_dim
            if self.emotion_prototypes is None:
                self.emotion_prototypes = torch.zeros(self.num_classes, resolved_emotion_dim, device=device)
            else:
                if self.emotion_prototypes.shape[1] != resolved_emotion_dim:
                    self.emotion_prototypes = torch.zeros(
                        self.num_classes,
                        resolved_emotion_dim,
                        device=device,
                    )
                elif self.emotion_prototypes.device != device:
                    self.emotion_prototypes = self.emotion_prototypes.to(device)
        else:
            if emotion_dim is not None:
                self.emotion_prototypes = None
                self.emotion_dim = 0
            elif self.emotion_prototypes is not None and self.emotion_prototypes.device != device:
                self.emotion_prototypes = self.emotion_prototypes.to(device)
        self.lexical_matrix = self.lexical_matrix.to(device)
        self._apply_pending_snapshot()

    def adjacency_matrix(self, device: Optional[torch.device] = None) -> torch.Tensor:
        base_device = device or self.device or torch.device("cpu")
        lexical = self.lexical_matrix.to(base_device)
        if self.cooccurrence is None:
            combined = lexical
        else:
            dynamic = self.cooccurrence.to(base_device).clamp_min(0.0)
            row_sum = dynamic.sum(dim=1, keepdim=True)
            dynamic_norm = torch.where(
                row_sum > 0,
                dynamic / (row_sum + self.smoothing),
                torch.zeros_like(dynamic),
            )
            combined = self.lexical_weight * lexical + (1.0 - self.lexical_weight) * dynamic_norm
        row_sum = combined.sum(dim=1, keepdim=True)
        return torch.where(
            row_sum > 0,
            combined / (row_sum + self.smoothing),
            torch.full_like(combined, 1.0 / self.num_classes),
        )

    def compute_loss(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        emotion_features: Optional[torch.Tensor],
        config: NeuroSymbolicConfig,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if features.dim() != 2:
            features = features.view(features.size(0), -1)
        emotion_dim = None
        if emotion_features is not None and emotion_features.numel() > 0:
            if emotion_features.dim() == 1:
                emotion_features = emotion_features.unsqueeze(0)
            emotion_dim = emotion_features.shape[-1]
        self.ensure_buffers(features.shape[1], features.device, emotion_dim)
        adjacency = self.adjacency_matrix(features.device)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        temperature = max(1e-6, float(config.temperature))
        probs = torch.softmax(logits / temperature, dim=-1)
        neighbors = adjacency[labels]
        self_loop = float(min(max(config.self_loop, 0.0), 1.0))
        target_distribution = self_loop * one_hot + (1.0 - self_loop) * neighbors
        target_distribution = target_distribution / target_distribution.sum(dim=1, keepdim=True).clamp_min(1e-6)

        structural = (probs - target_distribution).pow(2).sum(dim=1)
        semantic = torch.zeros_like(structural)
        if self.concept_prototypes is not None:
            semantic_targets = target_distribution @ self.concept_prototypes
            semantic = (features - semantic_targets).pow(2).sum(dim=1)
        affective = torch.zeros_like(structural)
        if (
            emotion_features is not None
            and emotion_features.numel() > 0
            and self.emotion_prototypes is not None
        ):
            affective_targets = target_distribution @ self.emotion_prototypes
            affective = (emotion_features - affective_targets).pow(2).sum(dim=1)

        total = (
            config.structural_weight * structural
            + config.semantic_weight * semantic
            + config.affective_weight * affective
        )
        entropy = -(
            adjacency.clamp_min(1e-9) * adjacency.clamp_min(1e-9).log()
        ).sum(dim=1)
        summary = {
            "loss": float(total.detach().mean().item()),
            "structural": float(structural.detach().mean().item()),
            "semantic": float(semantic.detach().mean().item()),
            "affective": float(affective.detach().mean().item()),
            "cohesion": float(torch.diagonal(adjacency).mean().item()),
            "entropy": float(entropy.mean().item()),
        }
        return total, summary

    def update_state(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        emotion_features: Optional[torch.Tensor],
    ) -> None:
        if features.numel() == 0:
            return
        if features.dim() != 2:
            features = features.view(features.size(0), -1)
        emotion_dim = None
        if emotion_features is not None and emotion_features.numel() > 0:
            if emotion_features.dim() == 1:
                emotion_features = emotion_features.unsqueeze(0)
            emotion_dim = emotion_features.shape[-1]
        self.ensure_buffers(features.shape[1], features.device, emotion_dim)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            confidences, _ = probs.max(dim=1)
            confident_mask = confidences >= self.min_confidence
            if confident_mask.any() and self.cooccurrence is not None and self.graph_counts is not None:
                confident_probs = probs[confident_mask]
                confident_labels = labels[confident_mask]
                one_hot = F.one_hot(confident_labels, num_classes=self.num_classes).float()
                cooc_update = one_hot.T @ confident_probs
                self.cooccurrence = (
                    (1 - self.graph_momentum) * self.cooccurrence
                    + self.graph_momentum * cooc_update
                )
                count_update = one_hot.sum(dim=0)
                self.graph_counts = (
                    (1 - self.graph_momentum) * self.graph_counts
                    + self.graph_momentum * count_update
                )

            class_counts = torch.bincount(labels, minlength=self.num_classes).float().to(features.device)
            if class_counts.sum() > 0 and self.concept_prototypes is not None and self.concept_counts is not None:
                feature_sums = torch.zeros(self.num_classes, features.shape[1], device=features.device)
                feature_sums.index_add_(0, labels, features)
                for idx in range(self.num_classes):
                    count = float(class_counts[idx].item())
                    if count <= 0:
                        continue
                    mean_vec = feature_sums[idx] / max(count, 1.0)
                    if float(self.concept_counts[idx].item()) <= 0:
                        self.concept_prototypes[idx] = mean_vec
                        self.concept_counts[idx] = class_counts[idx]
                    else:
                        blend = self.feature_momentum
                        self.concept_prototypes[idx] = (
                            (1 - blend) * self.concept_prototypes[idx] + blend * mean_vec
                        )
                        self.concept_counts[idx] = (
                            (1 - blend) * self.concept_counts[idx] + blend * class_counts[idx]
                        )

            if (
                emotion_features is not None
                and emotion_features.numel() > 0
                and self.emotion_prototypes is not None
            ):
                emotion_sums = torch.zeros(self.num_classes, self.emotion_prototypes.shape[1], device=features.device)
                emotion_sums.index_add_(0, labels, emotion_features)
                for idx in range(self.num_classes):
                    count = float(class_counts[idx].item())
                    if count <= 0:
                        continue
                    mean_vec = emotion_sums[idx] / max(count, 1.0)
                    if float(self.emotion_prototypes[idx].abs().sum().item()) <= 0:
                        self.emotion_prototypes[idx] = mean_vec
                    else:
                        blend = self.feature_momentum
                        self.emotion_prototypes[idx] = (
                            (1 - blend) * self.emotion_prototypes[idx] + blend * mean_vec
                        )

            adjacency = self.adjacency_matrix(features.device)
            entropy = -(
                adjacency.clamp_min(1e-9) * adjacency.clamp_min(1e-9).log()
            ).sum(dim=1).mean()
            self.history.append(
                {
                    "confident": int(confident_mask.sum().item()),
                    "mean_confidence": float(confidences.mean().item()),
                    "graph_entropy": float(entropy.item()),
                }
            )
            self.total_updates += int(features.shape[0])

    def export_metadata(self) -> Dict[str, object]:
        adjacency = self.adjacency_matrix().detach().cpu().tolist()
        keyword_map = {
            self.idx_to_label[idx] if idx < len(self.idx_to_label) else str(idx): keywords
            for idx, keywords in enumerate(self.lexical_keywords)
        }
        return {
            "lexical_weight": self.lexical_weight,
            "graph_momentum": self.graph_momentum,
            "feature_momentum": self.feature_momentum,
            "min_confidence": self.min_confidence,
            "keywords": keyword_map,
            "history": list(self.history),
            "adjacency": adjacency,
            "total_updates": int(self.total_updates),
        }

    def export_metrics(self) -> Dict[str, float]:
        adjacency = self.adjacency_matrix()
        entropy = -(
            adjacency.clamp_min(1e-9) * adjacency.clamp_min(1e-9).log()
        ).sum(dim=1)
        diag_mean = torch.diagonal(adjacency).mean()
        return {
            "neuro_cohesion": float(diag_mean.item()),
            "neuro_entropy": float(entropy.mean().item()),
            "neuro_history": float(len(self.history)),
        }

    def snapshot(self) -> Dict[str, object]:
        return {
            "cooccurrence": self.cooccurrence.detach().cpu().tolist() if self.cooccurrence is not None else None,
            "graph_counts": self.graph_counts.detach().cpu().tolist() if self.graph_counts is not None else None,
            "concept_prototypes": self.concept_prototypes.detach().cpu().tolist() if self.concept_prototypes is not None else None,
            "concept_counts": self.concept_counts.detach().cpu().tolist() if self.concept_counts is not None else None,
            "emotion_prototypes": self.emotion_prototypes.detach().cpu().tolist() if self.emotion_prototypes is not None else None,
            "history": list(self.history),
            "total_updates": int(self.total_updates),
        }

    def load_snapshot(self, snapshot: Dict[str, object]) -> None:
        if not isinstance(snapshot, dict):
            return
        self._pending_snapshot = snapshot


class SelfDiscoveryOrchestrator:
    """Maintain counterfactual memories that sharpen generalisation."""

    def __init__(
        self,
        num_classes: int,
        *,
        feature_momentum: float = 0.25,
        counter_momentum: float = 0.3,
        imagination_momentum: float = 0.15,
        curiosity_weight: float = 0.5,
        history_limit: int = 128,
        smoothing: float = 1e-5,
    ) -> None:
        self.num_classes = int(max(1, num_classes))
        self.feature_momentum = float(max(1e-4, min(feature_momentum, 0.999)))
        self.counter_momentum = float(max(1e-4, min(counter_momentum, 0.999)))
        self.imagination_momentum = float(max(1e-4, min(imagination_momentum, 0.999)))
        self.curiosity_weight = float(max(0.0, curiosity_weight))
        self.smoothing = float(max(smoothing, 1e-6))
        self.history: deque[Dict[str, object]] = deque(maxlen=max(16, history_limit))
        self.feature_dim: Optional[int] = None
        self.emotion_dim: int = 0
        self.device: Optional[torch.device] = None
        self.positive_prototypes: Optional[torch.Tensor] = None
        self.positive_counts: Optional[torch.Tensor] = None
        self.counterfactual_prototypes: Optional[torch.Tensor] = None
        self.counterfactual_counts: Optional[torch.Tensor] = None
        self.expectation_trace: Optional[torch.Tensor] = None
        self.expectation_counts: Optional[torch.Tensor] = None
        self.curiosity_trace: Optional[torch.Tensor] = None
        self.emotion_positive: Optional[torch.Tensor] = None
        self.emotion_counter: Optional[torch.Tensor] = None
        self.total_updates: int = 0
        self._pending_snapshot: Optional[Dict[str, object]] = None

    def ensure_buffers(
        self,
        feature_dim: int,
        device: torch.device,
        emotion_dim: Optional[int] = None,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive for self-discovery orchestration")
        needs_init = self.positive_prototypes is None or self.positive_prototypes.shape[1] != feature_dim
        if needs_init:
            self.feature_dim = feature_dim
            self.device = device
            self.positive_prototypes = torch.zeros(self.num_classes, feature_dim, device=device)
            self.positive_counts = torch.zeros(self.num_classes, device=device)
            self.counterfactual_prototypes = torch.zeros(
                self.num_classes,
                self.num_classes,
                feature_dim,
                device=device,
            )
            self.counterfactual_counts = torch.zeros(self.num_classes, self.num_classes, device=device)
            self.expectation_trace = torch.zeros(self.num_classes, feature_dim, device=device)
            self.expectation_counts = torch.zeros(self.num_classes, device=device)
            self.curiosity_trace = torch.zeros(self.num_classes, device=device)
        elif self.positive_prototypes.device != device:
            self.positive_prototypes = self.positive_prototypes.to(device)
            if self.positive_counts is not None:
                self.positive_counts = self.positive_counts.to(device)
            if self.counterfactual_prototypes is not None:
                self.counterfactual_prototypes = self.counterfactual_prototypes.to(device)
            if self.counterfactual_counts is not None:
                self.counterfactual_counts = self.counterfactual_counts.to(device)
            if self.expectation_trace is not None:
                self.expectation_trace = self.expectation_trace.to(device)
            if self.expectation_counts is not None:
                self.expectation_counts = self.expectation_counts.to(device)
            if self.curiosity_trace is not None:
                self.curiosity_trace = self.curiosity_trace.to(device)
            if self.emotion_positive is not None:
                self.emotion_positive = self.emotion_positive.to(device)
            if self.emotion_counter is not None:
                self.emotion_counter = self.emotion_counter.to(device)
            self.device = device
        resolved_emotion_dim = int(emotion_dim or self.emotion_dim or 0)
        if resolved_emotion_dim > 0:
            if (
                self.emotion_positive is None
                or self.emotion_positive.shape[1] != resolved_emotion_dim
            ):
                self.emotion_positive = torch.zeros(
                    self.num_classes,
                    resolved_emotion_dim,
                    device=device,
                )
                self.emotion_counter = torch.zeros(
                    self.num_classes,
                    self.num_classes,
                    resolved_emotion_dim,
                    device=device,
                )
            elif self.emotion_positive.device != device:
                self.emotion_positive = self.emotion_positive.to(device)
                if self.emotion_counter is not None:
                    self.emotion_counter = self.emotion_counter.to(device)
            self.emotion_dim = resolved_emotion_dim
        self._apply_pending_snapshot()

    def _apply_pending_snapshot(self) -> None:
        if not self._pending_snapshot or self.device is None or self.feature_dim is None:
            return
        snapshot = self._pending_snapshot
        device = self.device
        try:
            positive = snapshot.get("positive_prototypes")
            if isinstance(positive, list) and self.positive_prototypes is not None:
                tensor = torch.tensor(positive, dtype=torch.float32, device=device)
                if tensor.shape == self.positive_prototypes.shape:
                    self.positive_prototypes.copy_(tensor)
            positive_counts = snapshot.get("positive_counts")
            if isinstance(positive_counts, list) and self.positive_counts is not None:
                tensor = torch.tensor(positive_counts, dtype=torch.float32, device=device)
                if tensor.shape == self.positive_counts.shape:
                    self.positive_counts.copy_(tensor)
            counter = snapshot.get("counterfactual_prototypes")
            if isinstance(counter, list) and self.counterfactual_prototypes is not None:
                tensor = torch.tensor(counter, dtype=torch.float32, device=device)
                if tensor.shape == self.counterfactual_prototypes.shape:
                    self.counterfactual_prototypes.copy_(tensor)
            counter_counts = snapshot.get("counterfactual_counts")
            if isinstance(counter_counts, list) and self.counterfactual_counts is not None:
                tensor = torch.tensor(counter_counts, dtype=torch.float32, device=device)
                if tensor.shape == self.counterfactual_counts.shape:
                    self.counterfactual_counts.copy_(tensor)
            expectation = snapshot.get("expectation_trace")
            if isinstance(expectation, list) and self.expectation_trace is not None:
                tensor = torch.tensor(expectation, dtype=torch.float32, device=device)
                if tensor.shape == self.expectation_trace.shape:
                    self.expectation_trace.copy_(tensor)
            expectation_counts = snapshot.get("expectation_counts")
            if isinstance(expectation_counts, list) and self.expectation_counts is not None:
                tensor = torch.tensor(expectation_counts, dtype=torch.float32, device=device)
                if tensor.shape == self.expectation_counts.shape:
                    self.expectation_counts.copy_(tensor)
            curiosity_trace = snapshot.get("curiosity_trace")
            if isinstance(curiosity_trace, list) and self.curiosity_trace is not None:
                tensor = torch.tensor(curiosity_trace, dtype=torch.float32, device=device)
                if tensor.shape == self.curiosity_trace.shape:
                    self.curiosity_trace.copy_(tensor)
            emotion_positive = snapshot.get("emotion_positive")
            if (
                isinstance(emotion_positive, list)
                and self.emotion_positive is not None
            ):
                tensor = torch.tensor(emotion_positive, dtype=torch.float32, device=device)
                if tensor.shape == self.emotion_positive.shape:
                    self.emotion_positive.copy_(tensor)
            emotion_counter = snapshot.get("emotion_counter")
            if isinstance(emotion_counter, list) and self.emotion_counter is not None:
                tensor = torch.tensor(emotion_counter, dtype=torch.float32, device=device)
                if tensor.shape == self.emotion_counter.shape:
                    self.emotion_counter.copy_(tensor)
        finally:
            self.total_updates = int(snapshot.get("total_updates", self.total_updates))
            history_items = snapshot.get("history", [])
            self.history.clear()
            if isinstance(history_items, list):
                for item in history_items:
                    if isinstance(item, dict):
                        cleaned = {
                            "confidence": float(item.get("confidence", 0.0)),
                            "curiosity": float(item.get("curiosity", 0.0)),
                            "counter_examples": int(item.get("counter_examples", 0)),
                        }
                        self.history.append(cleaned)
            self._pending_snapshot = None

    def compute_loss(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        config: SelfDiscoveryConfig,
        emotion_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if features.dim() != 2:
            features = features.view(features.size(0), -1)
        batch = features.size(0)
        if batch == 0:
            zero = torch.zeros(0, device=features.device, dtype=features.dtype)
            return zero, {
                "loss": 0.0,
                "alignment": 0.0,
                "contrast": 0.0,
                "imagination": 0.0,
                "emotion": 0.0,
                "confidence": 0.0,
                "curiosity": 0.0,
                "counter_share": 0.0,
            }
        if emotion_features is not None and emotion_features.dim() == 1:
            emotion_features = emotion_features.unsqueeze(0)
        emotion_dim = int(emotion_features.shape[-1]) if emotion_features is not None else None
        self.ensure_buffers(features.shape[1], features.device, emotion_dim)
        assert self.positive_prototypes is not None
        assert self.counterfactual_prototypes is not None
        assert self.positive_counts is not None
        assert self.counterfactual_counts is not None

        with torch.no_grad():
            positive_counts = self.positive_counts.detach()
            counter_counts = self.counterfactual_counts.detach()
            counter_total = (counter_counts > 0).float().mean().item()

        temperature = max(config.temperature, 1e-6)
        probs = torch.softmax(logits / temperature, dim=-1)
        confidence, predicted = probs.max(dim=1)
        positive = self.positive_prototypes.detach()
        counter = self.counterfactual_prototypes.detach()
        fallback_proto = probs @ positive
        available = (positive_counts[labels] > 0).unsqueeze(1)
        target_proto = torch.where(available, positive[labels], fallback_proto.detach())
        alignment = (features - target_proto).pow(2).sum(dim=1)

        counter_available = counter_counts[labels, predicted] > 0
        counter_proto = counter[labels, predicted]
        counter_proto = torch.where(
            counter_available.unsqueeze(1),
            counter_proto,
            target_proto,
        )
        pos_sim = F.cosine_similarity(features, target_proto, dim=-1, eps=1e-6)
        neg_sim = F.cosine_similarity(features, counter_proto, dim=-1, eps=1e-6)
        margin = max(0.0, config.margin)
        contrast = torch.relu(neg_sim - pos_sim + margin)
        contrast = contrast * counter_available.float()

        imagination = (fallback_proto - features).pow(2).sum(dim=1)

        emotion_loss = torch.zeros_like(alignment)
        if (
            emotion_features is not None
            and emotion_features.numel() > 0
            and self.emotion_positive is not None
        ):
            emotion_positive = self.emotion_positive.detach()
            target_emotion = emotion_positive[labels]
            emotion_available = (target_emotion.abs().sum(dim=1) > 0).unsqueeze(1)
            active_target = torch.where(
                emotion_available,
                target_emotion,
                torch.zeros_like(target_emotion),
            )
            emotion_loss = (emotion_features - active_target).pow(2).sum(dim=1)
            emotion_loss = emotion_loss * emotion_available.float().squeeze(1)

        curiosity = torch.relu(config.min_confidence - confidence)
        total = (
            config.alignment_weight * alignment
            + config.contrast_weight * contrast
            + config.imagination_weight * imagination
            + config.emotion_weight * emotion_loss
            + self.curiosity_weight * curiosity
        )

        summary = {
            "loss": float(total.detach().mean().item()),
            "alignment": float(alignment.detach().mean().item()),
            "contrast": float(contrast.detach().mean().item()),
            "imagination": float(imagination.detach().mean().item()),
            "emotion": float(emotion_loss.detach().mean().item()),
            "confidence": float(confidence.detach().mean().item()),
            "curiosity": float(curiosity.detach().mean().item()),
            "counter_share": float(counter_total),
        }
        return total, summary

    def update_state(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        config: SelfDiscoveryConfig,
        emotion_features: Optional[torch.Tensor] = None,
    ) -> None:
        if features.numel() == 0:
            return
        if features.dim() != 2:
            features = features.view(features.size(0), -1)
        emotion_dim = int(emotion_features.shape[-1]) if emotion_features is not None else None
        if emotion_features is not None and emotion_features.dim() == 1:
            emotion_features = emotion_features.unsqueeze(0)
        self.ensure_buffers(features.shape[1], features.device, emotion_dim)
        assert self.positive_prototypes is not None
        assert self.positive_counts is not None
        assert self.counterfactual_prototypes is not None
        assert self.counterfactual_counts is not None
        if emotion_features is not None and self.emotion_positive is not None:
            emotion_features = emotion_features.to(features.device)

        with torch.no_grad():
            temperature = max(config.temperature, 1e-6)
            probs = torch.softmax(logits / temperature, dim=-1)
            confidence, predicted = probs.max(dim=1)
            class_counts = torch.bincount(labels, minlength=self.num_classes).float().to(features.device)
            if class_counts.sum() > 0:
                feature_sums = torch.zeros(self.num_classes, features.shape[1], device=features.device)
                feature_sums.index_add_(0, labels, features)
                for idx in range(self.num_classes):
                    count = float(class_counts[idx].item())
                    if count <= 0:
                        continue
                    mean_vec = feature_sums[idx] / max(count, 1.0)
                    if float(self.positive_counts[idx].item()) <= 0:
                        self.positive_prototypes[idx] = mean_vec
                        self.positive_counts[idx] = class_counts[idx]
                    else:
                        blend = self.feature_momentum
                        self.positive_prototypes[idx] = (
                            (1 - blend) * self.positive_prototypes[idx] + blend * mean_vec
                        )
                        self.positive_counts[idx] = (
                            (1 - blend) * self.positive_counts[idx] + blend * class_counts[idx]
                        )

            needs_counter = (predicted != labels) | (confidence < config.min_confidence)
            if needs_counter.any():
                hard_features = features[needs_counter]
                hard_labels = labels[needs_counter]
                hard_pred = predicted[needs_counter]
                for feat_vec, true_idx, pred_idx in zip(hard_features, hard_labels, hard_pred):
                    true_i = int(true_idx.item())
                    pred_i = int(pred_idx.item())
                    proto = self.counterfactual_prototypes[true_i, pred_i]
                    if float(self.counterfactual_counts[true_i, pred_i].item()) <= 0:
                        self.counterfactual_prototypes[true_i, pred_i] = feat_vec
                    else:
                        blend = self.counter_momentum
                        self.counterfactual_prototypes[true_i, pred_i] = (
                            (1 - blend) * proto + blend * feat_vec
                        )
                    self.counterfactual_counts[true_i, pred_i] = self.counterfactual_counts[true_i, pred_i] + 1.0

            if (
                emotion_features is not None
                and emotion_features.numel() > 0
                and self.emotion_positive is not None
            ):
                emotion_sums = torch.zeros(self.num_classes, emotion_features.shape[1], device=features.device)
                emotion_sums.index_add_(0, labels, emotion_features)
                for idx in range(self.num_classes):
                    count = float(class_counts[idx].item())
                    if count <= 0:
                        continue
                    mean_vec = emotion_sums[idx] / max(count, 1.0)
                    current = self.emotion_positive[idx]
                    if float(current.abs().sum().item()) <= 0:
                        self.emotion_positive[idx] = mean_vec
                    else:
                        blend = self.feature_momentum
                        self.emotion_positive[idx] = (
                            (1 - blend) * current + blend * mean_vec
                        )
                if needs_counter.any() and self.emotion_counter is not None:
                    hard_emotion = emotion_features[needs_counter]
                    for emo_vec, true_idx, pred_idx in zip(hard_emotion, hard_labels, hard_pred):
                        true_i = int(true_idx.item())
                        pred_i = int(pred_idx.item())
                        proto = self.emotion_counter[true_i, pred_i]
                        if float(proto.abs().sum().item()) <= 0:
                            self.emotion_counter[true_i, pred_i] = emo_vec
                        else:
                            blend = self.counter_momentum
                            self.emotion_counter[true_i, pred_i] = (
                                (1 - blend) * proto + blend * emo_vec
                            )

            if self.expectation_trace is not None and self.expectation_counts is not None:
                expected = probs.transpose(0, 1) @ features
                counts = probs.sum(dim=0)
                blend = self.imagination_momentum
                self.expectation_trace = (
                    (1 - blend) * self.expectation_trace + blend * expected
                )
                self.expectation_counts = (
                    (1 - blend) * self.expectation_counts + blend * counts
                )
            if self.curiosity_trace is not None:
                curiosity = torch.relu(config.min_confidence - confidence)
                curiosity_sum = torch.zeros(self.num_classes, device=features.device)
                curiosity_sum.index_add_(0, labels, curiosity)
                blend = self.imagination_momentum
                self.curiosity_trace = (
                    (1 - blend) * self.curiosity_trace + blend * curiosity_sum
                )

            self.history.append(
                {
                    "confidence": float(confidence.mean().item()),
                    "curiosity": float(torch.relu(config.min_confidence - confidence).mean().item()),
                    "counter_examples": int(needs_counter.sum().item()),
                }
            )
            self.total_updates += int(features.shape[0])

    def snapshot(self) -> Dict[str, object]:
        return {
            "positive_prototypes": self.positive_prototypes.detach().cpu().tolist()
            if self.positive_prototypes is not None
            else None,
            "positive_counts": self.positive_counts.detach().cpu().tolist()
            if self.positive_counts is not None
            else None,
            "counterfactual_prototypes": self.counterfactual_prototypes.detach().cpu().tolist()
            if self.counterfactual_prototypes is not None
            else None,
            "counterfactual_counts": self.counterfactual_counts.detach().cpu().tolist()
            if self.counterfactual_counts is not None
            else None,
            "expectation_trace": self.expectation_trace.detach().cpu().tolist()
            if self.expectation_trace is not None
            else None,
            "expectation_counts": self.expectation_counts.detach().cpu().tolist()
            if self.expectation_counts is not None
            else None,
            "curiosity_trace": self.curiosity_trace.detach().cpu().tolist()
            if self.curiosity_trace is not None
            else None,
            "emotion_positive": self.emotion_positive.detach().cpu().tolist()
            if self.emotion_positive is not None
            else None,
            "emotion_counter": self.emotion_counter.detach().cpu().tolist()
            if self.emotion_counter is not None
            else None,
            "history": list(self.history),
            "total_updates": int(self.total_updates),
        }

    def load_snapshot(self, snapshot: Dict[str, object]) -> None:
        if not isinstance(snapshot, dict):
            return
        self._pending_snapshot = snapshot
        if self.device is not None and self.feature_dim is not None:
            self._apply_pending_snapshot()

    def export_metadata(self) -> Dict[str, object]:
        coverage = 0.0
        counter_coverage = 0.0
        if self.positive_counts is not None and self.positive_counts.numel() > 0:
            coverage = float(((self.positive_counts > 0).float().mean()).item())
        if self.counterfactual_counts is not None and self.counterfactual_counts.numel() > 0:
            counter_coverage = float(((self.counterfactual_counts > 0).float().mean()).item())
        return {
            "feature_momentum": self.feature_momentum,
            "counter_momentum": self.counter_momentum,
            "imagination_momentum": self.imagination_momentum,
            "curiosity_weight": self.curiosity_weight,
            "smoothing": self.smoothing,
            "emotion_dim": self.emotion_dim,
            "prototype_coverage": coverage,
            "counter_coverage": counter_coverage,
            "history": list(self.history),
            "total_updates": int(self.total_updates),
        }


class TranscendentCognitionEngine:
    """Coordinate multi-modal, multi-hypothesis cognition over the latent space."""

    def __init__(
        self,
        num_classes: int,
        *,
        feature_momentum: float = 0.2,
        counter_momentum: float = 0.25,
        transition_momentum: float = 0.15,
        imagination_momentum: float = 0.1,
        history_limit: int = 128,
        max_glimpses: int = 4,
    ) -> None:
        self.num_classes = int(max(1, num_classes))
        self.feature_momentum = float(max(1e-4, min(feature_momentum, 0.999)))
        self.counter_momentum = float(max(1e-4, min(counter_momentum, 0.999)))
        self.transition_momentum = float(max(1e-4, min(transition_momentum, 0.999)))
        self.imagination_momentum = float(max(1e-4, min(imagination_momentum, 0.999)))
        self.max_glimpses = int(max(1, max_glimpses))
        self.history: deque[Dict[str, object]] = deque(maxlen=max(32, history_limit))
        self.feature_dim: Optional[int] = None
        self.emotion_dim: int = 0
        self.device: Optional[torch.device] = None
        self.feature_bank: Optional[torch.Tensor] = None
        self.counter_bank: Optional[torch.Tensor] = None
        self.imagination_bank: Optional[torch.Tensor] = None
        self.transition_matrix: Optional[torch.Tensor] = None
        self.emotion_bank: Optional[torch.Tensor] = None
        self.total_updates: int = 0
        self._pending_snapshot: Optional[Dict[str, object]] = None

    def _trim_distribution(self, distribution: torch.Tensor) -> torch.Tensor:
        if distribution.numel() == 0 or self.max_glimpses <= 0:
            return distribution
        if self.max_glimpses >= distribution.shape[1]:
            return distribution
        top_values, top_indices = torch.topk(distribution, k=self.max_glimpses, dim=1)
        trimmed = torch.zeros_like(distribution)
        trimmed.scatter_(1, top_indices, top_values)
        denom = trimmed.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return trimmed / denom

    def _apply_pending_snapshot(self) -> None:
        if not self._pending_snapshot:
            return
        if self.feature_bank is None or self.device is None or self.feature_dim is None:
            return
        snapshot = self._pending_snapshot
        device = self.device

        def _copy_tensor(attr: Optional[torch.Tensor], key: str) -> Optional[torch.Tensor]:
            if attr is None:
                return None
            data = snapshot.get(key)
            if not isinstance(data, list):
                return attr
            tensor = torch.tensor(data, dtype=torch.float32, device=device)
            if tensor.shape == attr.shape:
                attr.copy_(tensor)
            return attr

        self.feature_bank = _copy_tensor(self.feature_bank, "feature_bank")
        self.counter_bank = _copy_tensor(self.counter_bank, "counter_bank")
        self.imagination_bank = _copy_tensor(self.imagination_bank, "imagination_bank")
        if self.transition_matrix is not None:
            transition_data = snapshot.get("transition_matrix")
            if isinstance(transition_data, list):
                tensor = torch.tensor(transition_data, dtype=torch.float32, device=device)
                if tensor.shape == self.transition_matrix.shape:
                    self.transition_matrix.copy_(tensor)
        emotion_data = snapshot.get("emotion_bank")
        if isinstance(emotion_data, list):
            tensor = torch.tensor(emotion_data, dtype=torch.float32, device=device)
            if tensor.dim() == 2:
                self.emotion_dim = tensor.shape[1]
                if self.emotion_bank is None or self.emotion_bank.shape != tensor.shape:
                    self.emotion_bank = torch.zeros_like(tensor)
                self.emotion_bank.copy_(tensor)
        history_items = snapshot.get("history", [])
        if isinstance(history_items, list):
            for item in history_items:
                if isinstance(item, dict):
                    entry = {
                        "confidence": float(item.get("confidence", 0.0)),
                        "entropy": float(item.get("entropy", 0.0)),
                        "transition_coherence": float(item.get("transition_coherence", 0.0)),
                    }
                    self.history.append(entry)
        self.total_updates = int(snapshot.get("total_updates", self.total_updates))
        self._pending_snapshot = None

    def ensure_buffers(
        self,
        feature_dim: int,
        device: torch.device,
        emotion_dim: Optional[int] = None,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive for transcendent cognition")
        needs_init = self.feature_bank is None or self.feature_bank.shape[1] != feature_dim
        if needs_init:
            self.feature_dim = feature_dim
            self.device = device
            self.feature_bank = torch.zeros(self.num_classes, feature_dim, device=device)
            self.counter_bank = torch.zeros(self.num_classes, feature_dim, device=device)
            self.imagination_bank = torch.zeros(self.num_classes, feature_dim, device=device)
        elif self.feature_bank.device != device:
            self.feature_bank = self.feature_bank.to(device)
            if self.counter_bank is not None:
                self.counter_bank = self.counter_bank.to(device)
            if self.imagination_bank is not None:
                self.imagination_bank = self.imagination_bank.to(device)
            self.device = device
        if self.transition_matrix is None or self.transition_matrix.shape != (self.num_classes, self.num_classes):
            self.transition_matrix = torch.eye(self.num_classes, device=device)
        elif self.transition_matrix.device != device:
            self.transition_matrix = self.transition_matrix.to(device)
        if emotion_dim is not None and emotion_dim > 0:
            if self.emotion_bank is None or self.emotion_bank.shape[1] != emotion_dim:
                self.emotion_dim = emotion_dim
                self.emotion_bank = torch.zeros(self.num_classes, emotion_dim, device=device)
            elif self.emotion_bank.device != device:
                self.emotion_bank = self.emotion_bank.to(device)
        self._apply_pending_snapshot()

    def normalised_transition(self, *, device: Optional[torch.device] = None) -> torch.Tensor:
        if self.transition_matrix is None:
            base = torch.eye(self.num_classes, device=device or self.device or torch.device("cpu"))
        else:
            base = self.transition_matrix
            if device is not None and base.device != device:
                base = base.to(device)
        sums = base.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return base / sums

    def compute_loss(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        config: TranscendentCognitionConfig,
        *,
        emotion_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch = features.shape[0]
        if batch == 0:
            zero = torch.zeros(0, device=features.device, dtype=features.dtype)
            return zero, {
                "loss": 0.0,
                "stability": 0.0,
                "divergence": 0.0,
                "foresight": 0.0,
                "synthesis": 0.0,
                "affective": 0.0,
                "entropy": 0.0,
                "coherence": 0.0,
                "samples": 0,
            }

        emotion_dim = None
        if emotion_features is not None and emotion_features.numel() > 0 and emotion_features.dim() == 2:
            emotion_dim = emotion_features.shape[1]
        self.ensure_buffers(features.shape[1], features.device, emotion_dim)
        assert self.feature_bank is not None and self.counter_bank is not None

        temperature = max(1e-6, float(config.temperature))
        distribution = torch.softmax(logits / temperature, dim=-1)
        distribution = self._trim_distribution(distribution)

        prototypes = self.feature_bank.detach()
        counter = self.counter_bank.detach()
        combined = prototypes + counter
        target_proto = prototypes[targets]
        stability = (features - target_proto).pow(2).sum(dim=1)

        diff = features.unsqueeze(1) - combined.unsqueeze(0)
        diff_sq = diff.pow(2).sum(dim=-1)
        one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).bool()
        masked_diff = diff_sq.masked_fill(one_hot, 0.0)
        masked_dist = distribution * (~one_hot).float()
        denom = masked_dist.sum(dim=1, keepdim=True).clamp_min(1e-6)
        normalised_masked_dist = masked_dist / denom
        if config.margin > 0:
            divergence_term = torch.relu(config.margin - masked_diff)
        else:
            divergence_term = masked_diff
        divergence = (normalised_masked_dist * divergence_term).sum(dim=1)

        expected_combined = distribution @ combined
        foresight = (features - expected_combined).pow(2).sum(dim=1)

        transition = self.normalised_transition(device=features.device)
        target_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
        future_weights = target_one_hot @ transition
        future_weights = future_weights / future_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        synthesis_vec = future_weights @ combined
        synthesis = (synthesis_vec - target_proto).pow(2).sum(dim=1)

        affective = torch.zeros_like(stability)
        if (
            emotion_features is not None
            and emotion_features.numel() > 0
            and self.emotion_bank is not None
        ):
            emotion_bank = self.emotion_bank.detach()
            expected_emotion = distribution @ emotion_bank
            affective = (emotion_features - expected_emotion).pow(2).sum(dim=1)

        entropy = -(distribution.clamp_min(1e-9).log() * distribution).sum(dim=1)

        total = (
            config.stability_weight * stability
            + config.divergence_weight * divergence
            + config.foresight_weight * foresight
            + config.synthesis_weight * synthesis
            + config.affective_weight * affective
            + config.entropy_weight * entropy
        )
        total = total / temperature

        coherence = float(torch.diagonal(transition).mean().item())

        summary = {
            "loss": float(total.detach().mean().item()),
            "stability": float(stability.detach().mean().item()),
            "divergence": float(divergence.detach().mean().item()),
            "foresight": float(foresight.detach().mean().item()),
            "synthesis": float(synthesis.detach().mean().item()),
            "affective": float(affective.detach().mean().item()),
            "entropy": float(entropy.detach().mean().item()),
            "coherence": coherence,
            "samples": batch,
        }
        return total, summary

    def update_state(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        config: TranscendentCognitionConfig,
        emotion_features: Optional[torch.Tensor] = None,
    ) -> None:
        if features.numel() == 0:
            return
        emotion_dim = None
        if emotion_features is not None and emotion_features.numel() > 0 and emotion_features.dim() == 2:
            emotion_dim = emotion_features.shape[1]
        self.ensure_buffers(features.shape[1], features.device, emotion_dim)
        assert self.feature_bank is not None and self.counter_bank is not None

        with torch.no_grad():
            temperature = max(1e-6, float(config.temperature))
            distribution = torch.softmax(logits / temperature, dim=-1)
            distribution = self._trim_distribution(distribution)

            blend = self.feature_momentum
            for class_idx in targets.unique(sorted=False):
                idx = int(class_idx.item())
                mask = targets == class_idx
                if not mask.any():
                    continue
                class_mean = features[mask].mean(dim=0)
                self.feature_bank[idx] = (1 - blend) * self.feature_bank[idx] + blend * class_mean

            expected = distribution.t() @ features
            counts = distribution.sum(dim=0).unsqueeze(1)
            for idx in range(self.num_classes):
                count = float(counts[idx].item())
                if count <= 0:
                    continue
                target_mean = expected[idx] / max(count, 1e-6)
                update = target_mean - self.feature_bank[idx]
                c_blend = self.counter_momentum
                self.counter_bank[idx] = (1 - c_blend) * self.counter_bank[idx] + c_blend * update

            t_blend = self.transition_momentum
            for class_idx in targets.unique(sorted=False):
                idx = int(class_idx.item())
                mask = targets == class_idx
                if not mask.any():
                    continue
                mean_trans = distribution[mask].mean(dim=0)
                self.transition_matrix[idx] = (1 - t_blend) * self.transition_matrix[idx] + t_blend * mean_trans

            combined = self.feature_bank + self.counter_bank
            imagined = distribution @ combined
            i_blend = self.imagination_momentum
            for class_idx in targets.unique(sorted=False):
                idx = int(class_idx.item())
                mask = targets == class_idx
                if not mask.any():
                    continue
                imagined_mean = imagined[mask].mean(dim=0)
                self.imagination_bank[idx] = (1 - i_blend) * self.imagination_bank[idx] + i_blend * imagined_mean

            if (
                emotion_features is not None
                and emotion_features.numel() > 0
                and self.emotion_bank is not None
            ):
                emotion_expected = distribution.t() @ emotion_features
                emotion_counts = distribution.sum(dim=0).unsqueeze(1)
                for idx in range(self.num_classes):
                    count = float(emotion_counts[idx].item())
                    if count <= 0:
                        continue
                    update = emotion_expected[idx] / max(count, 1e-6)
                    e_blend = self.counter_momentum
                    self.emotion_bank[idx] = (1 - e_blend) * self.emotion_bank[idx] + e_blend * update

            entropy = -(distribution.clamp_min(1e-9).log() * distribution).sum(dim=1)
            confidence = distribution.max(dim=1).values
            coherence = float(torch.diagonal(self.normalised_transition()).mean().item())
            self.history.append(
                {
                    "confidence": float(confidence.mean().item()),
                    "entropy": float(entropy.mean().item()),
                    "transition_coherence": coherence,
                }
            )
            self.total_updates += int(features.shape[0])

    def export_metadata(self) -> Dict[str, object]:
        return {
            "feature_momentum": self.feature_momentum,
            "counter_momentum": self.counter_momentum,
            "transition_momentum": self.transition_momentum,
            "imagination_momentum": self.imagination_momentum,
            "max_glimpses": self.max_glimpses,
            "history_limit": self.history.maxlen,
            "history": list(self.history),
            "total_updates": int(self.total_updates),
        }

    def export_metrics(self) -> Dict[str, float]:
        transition = self.normalised_transition()
        diag = torch.diagonal(transition)
        metrics = {
            "transcendent_transition_coherence": float(diag.mean().item()),
            "transcendent_updates": float(self.total_updates),
            "transcendent_history": float(len(self.history)),
        }
        if self.imagination_bank is not None:
            metrics["transcendent_imagination_norm"] = float(
                self.imagination_bank.norm(dim=1).mean().item()
            )
        return metrics

    def snapshot(self) -> Dict[str, object]:
        return {
            "feature_bank": self.feature_bank.detach().cpu().tolist() if self.feature_bank is not None else None,
            "counter_bank": self.counter_bank.detach().cpu().tolist() if self.counter_bank is not None else None,
            "imagination_bank": self.imagination_bank.detach().cpu().tolist() if self.imagination_bank is not None else None,
            "transition_matrix": self.transition_matrix.detach().cpu().tolist() if self.transition_matrix is not None else None,
            "emotion_bank": self.emotion_bank.detach().cpu().tolist() if self.emotion_bank is not None else None,
            "history": list(self.history),
            "total_updates": int(self.total_updates),
        }

    def load_snapshot(self, snapshot: Dict[str, object]) -> None:
        if not isinstance(snapshot, dict):
            return
        self._pending_snapshot = snapshot
        self._apply_pending_snapshot()


class FrontierIntelligenceEngine:
    """Blend novelty scouting, abstraction, and curiosity-driven transfer."""

    def __init__(
        self,
        num_classes: int,
        *,
        concept_momentum: float = 0.2,
        bridge_momentum: float = 0.2,
        novelty_momentum: float = 0.25,
        meta_momentum: float = 0.1,
        emotion_momentum: float = 0.2,
        history_limit: int = 192,
    ) -> None:
        self.num_classes = int(max(1, num_classes))
        self.concept_momentum = float(max(1e-4, min(concept_momentum, 0.999)))
        self.bridge_momentum = float(max(1e-4, min(bridge_momentum, 0.999)))
        self.novelty_momentum = float(max(1e-4, min(novelty_momentum, 0.999)))
        self.meta_momentum = float(max(1e-4, min(meta_momentum, 0.999)))
        self.emotion_momentum = float(max(1e-4, min(emotion_momentum, 0.999)))
        self.history: deque[Dict[str, object]] = deque(maxlen=max(32, history_limit))
        self.feature_dim: Optional[int] = None
        self.emotion_dim: int = 0
        self.device: Optional[torch.device] = None
        self.concept_bank: Optional[torch.Tensor] = None
        self.bridge_bank: Optional[torch.Tensor] = None
        self.novelty_bank: Optional[torch.Tensor] = None
        self.meta_bias: Optional[torch.Tensor] = None
        self.emotion_bank: Optional[torch.Tensor] = None
        self.total_updates: int = 0
        self._pending_snapshot: Optional[Dict[str, object]] = None

    def ensure_buffers(
        self,
        feature_dim: int,
        device: torch.device,
        emotion_dim: Optional[int] = None,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive for frontier intelligence")
        needs_init = self.concept_bank is None or self.concept_bank.shape[1] != feature_dim
        if needs_init:
            self.feature_dim = feature_dim
            self.device = device
            self.concept_bank = torch.zeros(self.num_classes, feature_dim, device=device)
            self.bridge_bank = torch.zeros(self.num_classes, feature_dim, device=device)
            self.novelty_bank = torch.zeros(self.num_classes, feature_dim, device=device)
            self.meta_bias = torch.zeros(feature_dim, device=device)
        elif self.concept_bank.device != device:  # type: ignore[union-attr]
            self.concept_bank = self.concept_bank.to(device)
            if self.bridge_bank is not None:
                self.bridge_bank = self.bridge_bank.to(device)
            if self.novelty_bank is not None:
                self.novelty_bank = self.novelty_bank.to(device)
            if self.meta_bias is not None:
                self.meta_bias = self.meta_bias.to(device)
            self.device = device
        resolved_emotion_dim = int(emotion_dim or self.emotion_dim or 0)
        if resolved_emotion_dim > 0:
            if (
                self.emotion_bank is None
                or self.emotion_bank.shape[1] != resolved_emotion_dim
            ):
                self.emotion_bank = torch.zeros(
                    self.num_classes,
                    resolved_emotion_dim,
                    device=device,
                )
            elif self.emotion_bank.device != device:
                self.emotion_bank = self.emotion_bank.to(device)
            self.emotion_dim = resolved_emotion_dim
        self._apply_pending_snapshot()

    def _apply_pending_snapshot(self) -> None:
        if not self._pending_snapshot or self.device is None or self.feature_dim is None:
            return
        snapshot = self._pending_snapshot
        device = self.device

        def _copy_tensor(current: Optional[torch.Tensor], key: str) -> Optional[torch.Tensor]:
            data = snapshot.get(key)
            if current is None or not isinstance(data, list):
                return current
            tensor = torch.tensor(data, dtype=torch.float32, device=device)
            if tensor.shape == current.shape:
                current.copy_(tensor)
            return current

        self.concept_bank = _copy_tensor(self.concept_bank, "concept_bank")
        self.bridge_bank = _copy_tensor(self.bridge_bank, "bridge_bank")
        self.novelty_bank = _copy_tensor(self.novelty_bank, "novelty_bank")
        if self.meta_bias is not None:
            meta_data = snapshot.get("meta_bias")
            if isinstance(meta_data, list):
                tensor = torch.tensor(meta_data, dtype=torch.float32, device=device)
                if tensor.shape == self.meta_bias.shape:
                    self.meta_bias.copy_(tensor)
        if self.emotion_bank is not None:
            self.emotion_bank = _copy_tensor(self.emotion_bank, "emotion_bank")
        history_items = snapshot.get("history", [])
        if isinstance(history_items, list):
            for item in history_items:
                if isinstance(item, dict):
                    self.history.append(
                        {
                            "confidence": float(item.get("confidence", 0.0)),
                            "curiosity": float(item.get("curiosity", 0.0)),
                            "diversity": float(item.get("diversity", 0.0)),
                        }
                    )
        self.total_updates = int(snapshot.get("total_updates", self.total_updates))
        self._pending_snapshot = None

    def compute_loss(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        config: FrontierIntelligenceConfig,
        *,
        emotion_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch = features.shape[0]
        if batch == 0:
            zero = torch.zeros(1, device=features.device, dtype=features.dtype)
            return zero, {
                "loss": 0.0,
                "novelty": 0.0,
                "abstraction": 0.0,
                "transfer": 0.0,
                "curiosity": 0.0,
                "emotion": 0.0,
                "meta": 0.0,
                "diversity": 0.0,
                "samples": 0.0,
            }
        emotion_dim = None
        if emotion_features is not None and emotion_features.numel() > 0 and emotion_features.dim() == 2:
            emotion_dim = emotion_features.shape[1]
        self.ensure_buffers(features.shape[1], features.device, emotion_dim)
        assert self.concept_bank is not None and self.bridge_bank is not None
        assert self.novelty_bank is not None and self.meta_bias is not None

        temperature = max(1e-6, float(config.temperature))
        distribution = torch.softmax(logits / temperature, dim=-1)

        target_proto = self.concept_bank[targets]
        abstraction_target = distribution @ self.concept_bank
        transfer_target = distribution @ self.bridge_bank

        novelty_distance = torch.cdist(features, self.novelty_bank)
        novelty_gap = torch.relu(config.margin - novelty_distance.min(dim=1).values)
        abstraction_penalty = (features - abstraction_target).pow(2).sum(dim=1)
        transfer_penalty = (features - transfer_target).pow(2).sum(dim=1)
        meta_penalty = (features - self.meta_bias.unsqueeze(0)).pow(2).sum(dim=1)
        confidence = distribution.max(dim=1).values
        curiosity_penalty = torch.relu(config.margin - confidence)

        if (
            emotion_features is not None
            and emotion_features.numel() > 0
            and self.emotion_bank is not None
        ):
            predicted_emotion = distribution @ self.emotion_bank
            emotion_penalty = (emotion_features - predicted_emotion).pow(2).sum(dim=1)
        else:
            emotion_penalty = torch.zeros_like(curiosity_penalty)

        novelty_component = novelty_gap.mean()
        abstraction_component = abstraction_penalty.mean()
        transfer_component = transfer_penalty.mean()
        curiosity_component = curiosity_penalty.mean()
        emotion_component = emotion_penalty.mean()
        meta_component = meta_penalty.mean()

        total_loss = (
            config.novelty_weight * novelty_component
            + config.abstraction_weight * abstraction_component
            + config.transfer_weight * transfer_component
            + config.curiosity_weight * curiosity_component
            + config.emotion_weight * emotion_component
            + config.meta_weight * meta_component
        )

        summary = {
            "loss": float(total_loss.detach().item()),
            "novelty": float(novelty_component.detach().item()),
            "abstraction": float(abstraction_component.detach().item()),
            "transfer": float(transfer_component.detach().item()),
            "curiosity": float(curiosity_component.detach().item()),
            "emotion": float(emotion_component.detach().item()),
            "meta": float(meta_component.detach().item()),
            "diversity": float(novelty_distance.mean().detach().item()),
            "samples": float(batch),
        }
        return total_loss, summary

    def update_state(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        config: FrontierIntelligenceConfig,
        emotion_features: Optional[torch.Tensor] = None,
    ) -> None:
        if features.numel() == 0:
            return
        emotion_dim = None
        if emotion_features is not None and emotion_features.numel() > 0 and emotion_features.dim() == 2:
            emotion_dim = emotion_features.shape[1]
        self.ensure_buffers(features.shape[1], features.device, emotion_dim)
        assert self.concept_bank is not None and self.bridge_bank is not None
        assert self.novelty_bank is not None and self.meta_bias is not None

        with torch.no_grad():
            temperature = max(1e-6, float(config.temperature))
            distribution = torch.softmax(logits / temperature, dim=-1)
            confidence = distribution.max(dim=1).values
            curiosity = torch.relu(config.margin - confidence)

            for class_idx in targets.unique(sorted=False):
                idx = int(class_idx.item())
                mask = targets == class_idx
                if not mask.any():
                    continue
                class_mean = features[mask].mean(dim=0)
                blend = self.concept_momentum
                self.concept_bank[idx] = (1 - blend) * self.concept_bank[idx] + blend * class_mean

            expected = distribution.t() @ features
            counts = distribution.sum(dim=0).unsqueeze(1).clamp_min(1e-6)
            bridge_target = expected / counts
            blend = self.bridge_momentum
            self.bridge_bank = (1 - blend) * self.bridge_bank + blend * bridge_target

            novelty_weights = curiosity.unsqueeze(1) * distribution
            novelty_counts = novelty_weights.sum(dim=0).unsqueeze(1)
            if novelty_counts.gt(0).any():
                weighted = novelty_weights.t() @ features
                novelty_target = weighted / novelty_counts.clamp_min(1e-6)
                blend = self.novelty_momentum
                mask = novelty_counts.squeeze(1) > 0
                if mask.any():
                    current = self.novelty_bank[mask]
                    updates = novelty_target[mask]
                    self.novelty_bank[mask] = (1 - blend) * current + blend * updates

            meta_blend = self.meta_momentum
            self.meta_bias = (1 - meta_blend) * self.meta_bias + meta_blend * features.mean(dim=0)

            if (
                emotion_features is not None
                and emotion_features.numel() > 0
                and self.emotion_bank is not None
            ):
                emotion_expected = distribution.t() @ emotion_features
                emotion_counts = distribution.sum(dim=0).unsqueeze(1).clamp_min(1e-6)
                target = emotion_expected / emotion_counts
                blend = self.emotion_momentum
                self.emotion_bank = (1 - blend) * self.emotion_bank + blend * target

            diversity_score = float(torch.cdist(self.concept_bank, self.novelty_bank).mean().item())
            self.history.append(
                {
                    "confidence": float(confidence.mean().item()),
                    "curiosity": float(curiosity.mean().item()),
                    "diversity": diversity_score,
                }
            )
            self.total_updates += int(features.shape[0])

    def export_metadata(self) -> Dict[str, object]:
        return {
            "concept_momentum": self.concept_momentum,
            "bridge_momentum": self.bridge_momentum,
            "novelty_momentum": self.novelty_momentum,
            "meta_momentum": self.meta_momentum,
            "emotion_momentum": self.emotion_momentum,
            "history_limit": self.history.maxlen,
            "history": list(self.history),
            "total_updates": int(self.total_updates),
        }

    def export_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "frontier_updates": float(self.total_updates),
            "frontier_history": float(len(self.history)),
        }
        if self.bridge_bank is not None:
            metrics["frontier_bridge_norm"] = float(self.bridge_bank.norm(dim=1).mean().item())
        if self.novelty_bank is not None and self.concept_bank is not None:
            metrics["frontier_diversity_span"] = float(
                torch.cdist(self.concept_bank, self.novelty_bank).mean().item()
            )
        return metrics

    def snapshot(self) -> Dict[str, object]:
        return {
            "concept_bank": self.concept_bank.detach().cpu().tolist() if self.concept_bank is not None else None,
            "bridge_bank": self.bridge_bank.detach().cpu().tolist() if self.bridge_bank is not None else None,
            "novelty_bank": self.novelty_bank.detach().cpu().tolist() if self.novelty_bank is not None else None,
            "meta_bias": self.meta_bias.detach().cpu().tolist() if self.meta_bias is not None else None,
            "emotion_bank": self.emotion_bank.detach().cpu().tolist() if self.emotion_bank is not None else None,
            "history": list(self.history),
            "total_updates": int(self.total_updates),
        }

    def load_snapshot(self, snapshot: Dict[str, object]) -> None:
        if not isinstance(snapshot, dict):
            return
        self._pending_snapshot = snapshot
        if self.device is not None and self.feature_dim is not None:
            self._apply_pending_snapshot()


@dataclass
class EncodedExample:
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    label: int
    weight: float
    teacher_logits: torch.Tensor
    emotion_vector: torch.Tensor


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
    self_play_rounds: List[Dict[str, object]]
    total_self_play_added: int
    self_play_examples: List[Tuple[str, str, float]]
    model_state: Dict[str, torch.Tensor]
    evaluation_outputs: List[Dict[str, Union[str, float]]]
    run_tag_suffix: Optional[str]


@dataclass
class DistillationConfig:
    alpha: float
    temperature: float


@dataclass
class CurriculumSample:
    base_weight: float
    weight: float
    multiplier: float


class AdaptiveCurriculum:
    """Track per-example difficulty and amplify hard examples adaptively."""

    def __init__(
        self,
        *,
        start_epoch: int,
        momentum: float,
        min_multiplier: float,
        max_multiplier: float,
        hard_boost: float,
        difficulty_power: float,
        history_limit: int = 200,
    ) -> None:
        self.start_epoch = max(0, int(start_epoch))
        self.momentum = float(max(0.0, min(momentum, 0.999)))
        self.min_multiplier = max(0.0, min_multiplier)
        self.max_multiplier = max(self.min_multiplier + 1e-6, max_multiplier)
        self.hard_boost = max(0.0, hard_boost)
        self.difficulty_power = max(0.5, difficulty_power)
        self.samples: Dict[str, CurriculumSample] = {}
        self.history: List[Dict[str, object]] = []
        self.history_limit = max(10, history_limit)

    def register_samples(self, texts: Sequence[str], weights: Sequence[float]) -> None:
        for text, weight in zip(texts, weights):
            base = float(weight) if float(weight) > 0 else 1.0
            existing = self.samples.get(text)
            if existing is None:
                self.samples[text] = CurriculumSample(
                    base_weight=base,
                    weight=float(weight),
                    multiplier=float(weight) / base if base else 1.0,
                )
            else:
                existing.weight = float(weight)
                if existing.base_weight <= 0:
                    existing.base_weight = base
                existing.multiplier = (
                    existing.weight / existing.base_weight
                    if existing.base_weight
                    else 1.0
                )

    def _bounded_weight(self, base_weight: float, target_weight: float) -> float:
        min_weight = base_weight * self.min_multiplier
        max_weight = base_weight * self.max_multiplier
        return float(min(max(target_weight, min_weight), max_weight))

    def update_difficulties(
        self,
        *,
        epoch: int,
        stage: str,
        texts: Sequence[str],
        labels: Sequence[str],
        weights: Sequence[float],
        targets: Sequence[int],
        probabilities: Sequence[Sequence[float]],
        idx_to_label: Sequence[str],
        snippet_fn: Callable[[str], str],
    ) -> Optional[Dict[str, object]]:
        if epoch < self.start_epoch:
            return None
        adjusted: List[Tuple[float, float, float, str, str, float]] = []
        for text, label, current_weight, target_idx, probs in zip(
            texts, labels, weights, targets, probabilities
        ):
            sample = self.samples.get(text)
            if sample is None:
                base_weight = float(current_weight) if current_weight > 0 else 1.0
                sample = CurriculumSample(
                    base_weight=base_weight,
                    weight=float(current_weight),
                    multiplier=1.0,
                )
                self.samples[text] = sample
            base_weight = sample.base_weight if sample.base_weight > 0 else 1.0
            current = sample.weight
            previous_multiplier = current / base_weight if base_weight else 1.0
            try:
                true_probability = float(probs[target_idx])
            except (IndexError, TypeError):
                true_probability = 0.0
            true_probability = max(0.0, min(1.0, true_probability))
            difficulty = max(0.0, 1.0 - true_probability)
            target_multiplier = 1.0 + self.hard_boost * (difficulty ** self.difficulty_power)
            desired_weight = base_weight * target_multiplier
            blended = (
                self.momentum * current
                + (1.0 - self.momentum) * desired_weight
            )
            bounded = self._bounded_weight(base_weight, blended)
            sample.weight = bounded
            sample.multiplier = bounded / base_weight if base_weight else 1.0
            adjusted.append(
                (
                    difficulty,
                    sample.multiplier,
                    previous_multiplier,
                    text,
                    label if label in idx_to_label else idx_to_label[target_idx],
                    true_probability,
                )
            )

        if not adjusted:
            return None

        boosted = sum(1 for _difficulty, mult, prev, *_ in adjusted if mult > prev + 1e-6)
        dampened = sum(1 for _difficulty, mult, prev, *_ in adjusted if mult < prev - 1e-6)
        mean_multiplier = float(sum(mult for _, mult, *_ in adjusted) / len(adjusted))
        max_multiplier = max(mult for _, mult, *_ in adjusted)
        min_multiplier = min(mult for _, mult, *_ in adjusted)
        hardest = sorted(adjusted, key=lambda item: item[0], reverse=True)[:3]
        hardest_summary = [
            {
                "text": snippet_fn(text),
                "label": label,
                "confidence": prob,
                "multiplier": mult,
            }
            for difficulty, mult, _prev, text, label, prob in hardest
        ]
        entry = {
            "epoch": float(epoch),
            "stage": stage,
            "boosted": int(boosted),
            "dampened": int(dampened),
            "examples": len(adjusted),
            "avg_multiplier": mean_multiplier,
            "max_multiplier": max_multiplier,
            "min_multiplier": min_multiplier,
            "hardest_examples": hardest_summary,
        }
        self.history.append(entry)
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit :]
        return entry

    def apply(self, texts: Sequence[str], weights: List[float]) -> None:
        for idx, text in enumerate(texts):
            sample = self.samples.get(text)
            if sample is None:
                continue
            weights[idx] = sample.weight

    def export_metadata(self) -> Dict[str, object]:
        return {
            "enabled": True,
            "start_epoch": self.start_epoch,
            "momentum": self.momentum,
            "min_multiplier": self.min_multiplier,
            "max_multiplier": self.max_multiplier,
            "hard_boost": self.hard_boost,
            "difficulty_power": self.difficulty_power,
            "updates": self.history,
        }

    def export_metrics(self) -> Dict[str, object]:
        if not self.history:
            return {
                "curriculum_updates": 0,
                "curriculum_avg_multiplier": 1.0,
                "curriculum_max_multiplier": 1.0,
                "curriculum_min_multiplier": 1.0,
            }
        avg_multiplier = mean(entry["avg_multiplier"] for entry in self.history)
        max_multiplier = max(entry["max_multiplier"] for entry in self.history)
        min_multiplier = min(entry["min_multiplier"] for entry in self.history)
        total_updates = len(self.history)
        total_boosted = sum(entry.get("boosted", 0) for entry in self.history)
        total_dampened = sum(entry.get("dampened", 0) for entry in self.history)
        return {
            "curriculum_updates": int(total_updates),
            "curriculum_avg_multiplier": float(avg_multiplier),
            "curriculum_max_multiplier": float(max_multiplier),
            "curriculum_min_multiplier": float(min_multiplier),
            "curriculum_total_boosted": int(total_boosted),
            "curriculum_total_dampened": int(total_dampened),
        }
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
        emotion_vectors: Optional[Sequence[Sequence[float]]] = None,
        emotion_encoder: Optional[Callable[[str], Sequence[float]]] = None,
        emotion_dim: Optional[int] = None,
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
        resolved_emotion_dim = int(emotion_dim or 0)
        include_emotion = resolved_emotion_dim > 0
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
            raw_emotion: Optional[Sequence[float]] = None
            if emotion_vectors is not None:
                idx = len(self.examples)
                if idx < len(emotion_vectors):
                    raw_emotion = emotion_vectors[idx]
            elif emotion_encoder is not None:
                raw_emotion = emotion_encoder(text)
            if raw_emotion is not None:
                if resolved_emotion_dim == 0:
                    resolved_emotion_dim = len(raw_emotion)
                    include_emotion = resolved_emotion_dim > 0
                values = list(raw_emotion)
                if len(values) < resolved_emotion_dim:
                    values = values + [0.0] * (resolved_emotion_dim - len(values))
                elif len(values) > resolved_emotion_dim:
                    values = values[:resolved_emotion_dim]
                emotion_tensor = torch.tensor(values, dtype=torch.float32)
            elif include_emotion:
                emotion_tensor = torch.zeros(resolved_emotion_dim, dtype=torch.float32)
            else:
                emotion_tensor = torch.empty(0, dtype=torch.float32)
            self.examples.append(
                EncodedExample(
                    tokens=token_tensor,
                    attention_mask=mask_tensor,
                    label=label_id,
                    weight=float(weight),
                    teacher_logits=teacher_tensor,
                    emotion_vector=emotion_tensor,
                )
            )
        self.include_emotion = include_emotion
        self.emotion_dim = resolved_emotion_dim if include_emotion else 0

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        if self.include_emotion:
            return (
                example.tokens,
                torch.tensor(example.label, dtype=torch.long),
                torch.tensor(example.weight, dtype=torch.float32),
                example.attention_mask,
                example.teacher_logits,
                example.emotion_vector,
            )
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

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        if return_features:
            return logits, representation
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

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            output_hidden_states=return_features,
            return_dict=True,
        )
        logits = outputs.logits
        if return_features:
            hidden = None
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states:
                last_hidden = hidden_states[-1]
                if last_hidden.dim() >= 2:
                    hidden = last_hidden[:, 0]
            if hidden is None:
                hidden = logits
            return logits, hidden
        return logits


class SentenceTransformerClassifier(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_layer = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden = self.input_layer(inputs)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        logits = self.output_layer(hidden)
        if return_features:
            return logits, hidden
        return logits


class EmotionallyAdaptiveModel(nn.Module):
    """Wrap a base classifier with an emotion-aware fusion head."""

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        num_emotions: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_classes = int(num_classes)
        self.num_emotions = int(max(0, num_emotions))
        fusion_hidden = max(self.num_emotions * 2, self.num_classes, 16)
        self.emotion_transform = nn.Sequential(
            nn.Linear(self.num_emotions, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, self.num_classes),
        ) if self.num_emotions > 0 else None
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.num_classes * 2, self.num_classes),
            nn.GELU(),
            nn.Linear(self.num_classes, self.num_classes),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.5))
        self.supports_emotion_features = self.num_emotions > 0

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        emotion_features: Optional[torch.Tensor] = None,
        return_components: bool = False,
        return_features: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        base_features: Optional[torch.Tensor] = None
        if return_features:
            try:
                base_logits, base_features = self.base_model(
                    inputs,
                    attention_mask=attention_mask,
                    return_features=True,
                )
            except TypeError:
                base_logits = self.base_model(inputs, attention_mask=attention_mask)
        else:
            base_logits = self.base_model(inputs, attention_mask=attention_mask)
        if return_features and base_features is None:
            base_features = base_logits
        emotion_logits = None
        fused_logits = base_logits
        if (
            self.supports_emotion_features
            and emotion_features is not None
            and emotion_features.numel() > 0
            and self.emotion_transform is not None
        ):
            if emotion_features.dim() == 1:
                emotion_features = emotion_features.unsqueeze(0)
            emotion_features = emotion_features.to(base_logits.dtype)
            if emotion_features.shape[-1] == self.num_emotions:
                emotion_logits = self.emotion_transform(emotion_features)
                combined = torch.cat([base_logits, emotion_logits], dim=-1)
                gate = self.fusion_gate(combined)
                fused_logits = base_logits + torch.sigmoid(self.residual_scale) * gate
        if return_components and return_features:
            if emotion_logits is None:
                emotion_logits = torch.zeros_like(base_logits)
            assert base_features is not None
            return fused_logits, base_logits, emotion_logits, base_features
        if return_components:
            if emotion_logits is None:
                emotion_logits = torch.zeros_like(base_logits)
            return fused_logits, base_logits, emotion_logits
        if return_features:
            assert base_features is not None
            return fused_logits, base_features
        return fused_logits

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
    emotion_config: Optional[EmotionTrainingConfig] = None,
    meta_config: Optional[MetaCognitiveConfig] = None,
    neuro_config: Optional[NeuroSymbolicConfig] = None,
    discovery_config: Optional[SelfDiscoveryConfig] = None,
    transcendent_config: Optional[TranscendentCognitionConfig] = None,
    frontier_config: Optional[FrontierIntelligenceConfig] = None,
) -> Tuple[float, float, Dict[str, float]]:
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

    emotion_alignment_total = 0.0
    emotion_batches = 0
    meta_enabled = meta_config is not None and meta_config.enabled
    meta_loss_total = 0.0
    meta_attr_total = 0.0
    meta_rep_total = 0.0
    meta_novel_total = 0.0
    meta_gap_total = 0.0
    meta_entropy_total = 0.0
    meta_sample_total = 0.0
    meta_coverage_total = 0.0
    meta_batches = 0
    neuro_enabled = neuro_config is not None and neuro_config.enabled
    neuro_loss_total = 0.0
    neuro_struct_total = 0.0
    neuro_semantic_total = 0.0
    neuro_affective_total = 0.0
    neuro_entropy_total = 0.0
    neuro_cohesion_total = 0.0
    neuro_sample_total = 0.0
    discovery_enabled = discovery_config is not None and discovery_config.enabled
    discovery_loss_total = 0.0
    discovery_alignment_total = 0.0
    discovery_contrast_total = 0.0
    discovery_imagination_total = 0.0
    discovery_emotion_total = 0.0
    discovery_confidence_total = 0.0
    discovery_curiosity_total = 0.0
    discovery_counter_share_total = 0.0
    discovery_sample_total = 0.0
    discovery_batches = 0

    transcendent_enabled = (
        transcendent_config is not None and transcendent_config.enabled
    )
    transcendent_loss_total = 0.0
    transcendent_stability_total = 0.0
    transcendent_divergence_total = 0.0
    transcendent_foresight_total = 0.0
    transcendent_synthesis_total = 0.0
    transcendent_affective_total = 0.0
    transcendent_entropy_total = 0.0
    transcendent_coherence_total = 0.0
    transcendent_sample_total = 0.0
    transcendent_batches = 0

    frontier_enabled = frontier_config is not None and frontier_config.enabled
    frontier_loss_total = 0.0
    frontier_novelty_total = 0.0
    frontier_abstraction_total = 0.0
    frontier_transfer_total = 0.0
    frontier_curiosity_total = 0.0
    frontier_emotion_total = 0.0
    frontier_meta_total = 0.0
    frontier_diversity_total = 0.0
    frontier_sample_total = 0.0
    frontier_batches = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        emotion_features = None
        if len(batch) == 6:
            inputs, targets, weights, attention_mask, teacher_logits, emotion_features = batch
        elif len(batch) == 5:
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
        if emotion_features is not None:
            emotion_features = emotion_features.to(device=device, dtype=torch.float32)
            if emotion_features.dim() == 1:
                emotion_features = emotion_features.unsqueeze(0)
        supports_emotion = (
            emotion_features is not None
            and emotion_features.numel() > 0
            and emotion_config is not None
            and emotion_config.enabled
            and getattr(model, "supports_emotion_features", False)
        )

        context = autocast_context(amp_enabled)

        with context:
            features: Optional[torch.Tensor] = None
            wants_features = (
                meta_enabled
                or neuro_enabled
                or discovery_enabled
                or transcendent_enabled
                or frontier_enabled
            )
            if supports_emotion:
                if wants_features:
                    outputs = model(
                        inputs,
                        attention_mask=attention_mask,
                        emotion_features=emotion_features,
                        return_components=True,
                        return_features=True,
                    )
                    logits, _, _, features = outputs
                else:
                    logits, _, _ = model(
                        inputs,
                        attention_mask=attention_mask,
                        emotion_features=emotion_features,
                        return_components=True,
                    )
            elif wants_features:
                logits, features = model(
                    inputs,
                    attention_mask=attention_mask,
                    return_features=True,
                )
            else:
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
            if supports_emotion and emotion_features is not None:
                alignment = emotion_config.memory.alignment_loss(
                    logits,
                    emotion_features,
                    temperature=emotion_config.temperature,
                )
                loss_values = loss_values + alignment * emotion_config.weight
                emotion_alignment_total += float(alignment.detach().mean().item())
                emotion_batches += 1
            meta_summary: Optional[Dict[str, float]] = None
            neuro_summary: Optional[Dict[str, float]] = None
            discovery_summary: Optional[Dict[str, float]] = None
            transcendent_summary: Optional[Dict[str, float]] = None
            frontier_summary: Optional[Dict[str, float]] = None
            if meta_enabled and features is not None:
                regulariser, meta_summary = meta_config.introspector.compute_regulariser(
                    features,
                    targets,
                    logits,
                    meta_config,
                )
                loss_values = loss_values + regulariser
            if neuro_enabled and features is not None:
                ns_loss, neuro_summary = neuro_config.reasoner.compute_loss(
                    features,
                    logits,
                    targets,
                    emotion_features=emotion_features,
                    config=neuro_config,
                )
                loss_values = loss_values + ns_loss
            if discovery_enabled and features is not None:
                discovery_loss, discovery_summary = discovery_config.orchestrator.compute_loss(
                    features,
                    logits,
                    targets,
                    config=discovery_config,
                    emotion_features=emotion_features,
                )
                loss_values = loss_values + discovery_loss
            if transcendent_enabled and features is not None:
                transcendent_loss, transcendent_summary = transcendent_config.architect.compute_loss(
                    features,
                    logits,
                    targets,
                    transcendent_config,
                    emotion_features=emotion_features,
                )
                loss_values = loss_values + transcendent_loss
            if frontier_enabled and features is not None:
                frontier_loss, frontier_summary = frontier_config.catalyst.compute_loss(
                    features,
                    logits,
                    targets,
                    frontier_config,
                    emotion_features=emotion_features,
                )
                loss_values = loss_values + frontier_loss
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

        if meta_enabled and features is not None:
            meta_config.introspector.update_memory(
                features.detach(),
                targets.detach(),
                logits.detach(),
            )
            if meta_summary is not None:
                samples = float(meta_summary.get("samples", float(targets.size(0))))
                meta_sample_total += samples
                meta_loss_total += meta_summary.get("loss", 0.0) * samples
                meta_attr_total += meta_summary.get("attraction", 0.0) * samples
                meta_rep_total += meta_summary.get("repulsion", 0.0) * samples
                meta_novel_total += meta_summary.get("novelty", 0.0) * samples
                meta_gap_total += meta_summary.get("gap", 0.0) * samples
                meta_entropy_total += meta_summary.get("entropy", 0.0) * samples
                meta_coverage_total += meta_summary.get("coverage", 0.0)
                meta_batches += 1

        if neuro_enabled and features is not None:
            detached_emotion: Optional[torch.Tensor]
            if emotion_features is not None and emotion_features.numel() > 0:
                detached_emotion = emotion_features.detach()
            else:
                detached_emotion = None
            neuro_config.reasoner.update_state(
                features.detach(),
                logits.detach(),
                targets.detach(),
                emotion_features=detached_emotion,
            )
            if neuro_summary is not None:
                sample_count = float(targets.size(0))
                neuro_sample_total += sample_count
                neuro_loss_total += neuro_summary.get("loss", 0.0) * sample_count
                neuro_struct_total += neuro_summary.get("structural", 0.0) * sample_count
                neuro_semantic_total += neuro_summary.get("semantic", 0.0) * sample_count
                neuro_affective_total += neuro_summary.get("affective", 0.0) * sample_count
                neuro_entropy_total += neuro_summary.get("entropy", 0.0) * sample_count
                neuro_cohesion_total += neuro_summary.get("cohesion", 0.0) * sample_count
        if discovery_enabled and features is not None:
            detached_emotion: Optional[torch.Tensor]
            if emotion_features is not None and emotion_features.numel() > 0:
                detached_emotion = emotion_features.detach()
            else:
                detached_emotion = None
            discovery_config.orchestrator.update_state(
                features.detach(),
                logits.detach(),
                targets.detach(),
                config=discovery_config,
                emotion_features=detached_emotion,
            )
            if discovery_summary is not None:
                sample_count = float(targets.size(0))
                discovery_sample_total += sample_count
                discovery_loss_total += discovery_summary.get("loss", 0.0) * sample_count
                discovery_alignment_total += discovery_summary.get("alignment", 0.0) * sample_count
                discovery_contrast_total += discovery_summary.get("contrast", 0.0) * sample_count
                discovery_imagination_total += discovery_summary.get("imagination", 0.0) * sample_count
                discovery_emotion_total += discovery_summary.get("emotion", 0.0) * sample_count
                discovery_confidence_total += discovery_summary.get("confidence", 0.0) * sample_count
                discovery_curiosity_total += discovery_summary.get("curiosity", 0.0) * sample_count
                discovery_counter_share_total += discovery_summary.get("counter_share", 0.0)
                discovery_batches += 1
        if transcendent_enabled and features is not None:
            detached_emotion: Optional[torch.Tensor]
            if emotion_features is not None and emotion_features.numel() > 0:
                detached_emotion = emotion_features.detach()
            else:
                detached_emotion = None
            transcendent_config.architect.update_state(
                features.detach(),
                logits.detach(),
                targets.detach(),
                config=transcendent_config,
                emotion_features=detached_emotion,
            )
            if transcendent_summary is not None:
                samples = float(transcendent_summary.get("samples", float(targets.size(0))))
                transcendent_sample_total += samples
                transcendent_loss_total += transcendent_summary.get("loss", 0.0) * samples
                transcendent_stability_total += transcendent_summary.get("stability", 0.0) * samples
                transcendent_divergence_total += transcendent_summary.get("divergence", 0.0) * samples
                transcendent_foresight_total += transcendent_summary.get("foresight", 0.0) * samples
                transcendent_synthesis_total += transcendent_summary.get("synthesis", 0.0) * samples
                transcendent_affective_total += transcendent_summary.get("affective", 0.0) * samples
                transcendent_entropy_total += transcendent_summary.get("entropy", 0.0) * samples
                transcendent_coherence_total += transcendent_summary.get("coherence", 0.0)
                transcendent_batches += 1
        if frontier_enabled and features is not None:
            if emotion_features is not None and emotion_features.numel() > 0:
                detached_emotion = emotion_features.detach()
            else:
                detached_emotion = None
            frontier_config.catalyst.update_state(
                features.detach(),
                logits.detach(),
                targets.detach(),
                config=frontier_config,
                emotion_features=detached_emotion,
            )
            if frontier_summary is not None:
                samples = float(frontier_summary.get("samples", float(targets.size(0))))
                frontier_sample_total += samples
                frontier_loss_total += frontier_summary.get("loss", 0.0) * samples
                frontier_novelty_total += frontier_summary.get("novelty", 0.0) * samples
                frontier_abstraction_total += frontier_summary.get("abstraction", 0.0) * samples
                frontier_transfer_total += frontier_summary.get("transfer", 0.0) * samples
                frontier_curiosity_total += frontier_summary.get("curiosity", 0.0) * samples
                frontier_emotion_total += frontier_summary.get("emotion", 0.0) * samples
                frontier_meta_total += frontier_summary.get("meta", 0.0) * samples
                frontier_diversity_total += frontier_summary.get("diversity", 0.0)
                frontier_batches += 1

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
        {
            "optimizer_steps": optimizer_steps,
            "ema_updates": ema_updates,
            "swa_updates": swa_updates,
            "emotion_alignment": (emotion_alignment_total / emotion_batches) if emotion_batches else 0.0,
            "meta_loss": (meta_loss_total / meta_sample_total) if meta_sample_total else 0.0,
            "meta_attraction": (meta_attr_total / meta_sample_total) if meta_sample_total else 0.0,
            "meta_repulsion": (meta_rep_total / meta_sample_total) if meta_sample_total else 0.0,
            "meta_novelty": (meta_novel_total / meta_sample_total) if meta_sample_total else 0.0,
            "meta_gap": (meta_gap_total / meta_sample_total) if meta_sample_total else 0.0,
            "meta_entropy": (meta_entropy_total / meta_sample_total) if meta_sample_total else 0.0,
            "meta_coverage": (meta_coverage_total / meta_batches) if meta_batches else 0.0,
            "meta_updates": int(meta_config.introspector.total_updates) if meta_enabled else 0,
            "neuro_loss": (neuro_loss_total / neuro_sample_total) if neuro_sample_total else 0.0,
            "neuro_structural": (neuro_struct_total / neuro_sample_total) if neuro_sample_total else 0.0,
            "neuro_semantic": (neuro_semantic_total / neuro_sample_total) if neuro_sample_total else 0.0,
            "neuro_affective": (neuro_affective_total / neuro_sample_total) if neuro_sample_total else 0.0,
            "neuro_entropy": (neuro_entropy_total / neuro_sample_total) if neuro_sample_total else 0.0,
            "neuro_cohesion": (neuro_cohesion_total / neuro_sample_total) if neuro_sample_total else 0.0,
            "neuro_updates": int(neuro_config.reasoner.total_updates) if neuro_enabled else 0,
            "discovery_loss": (discovery_loss_total / discovery_sample_total) if discovery_sample_total else 0.0,
            "discovery_alignment": (discovery_alignment_total / discovery_sample_total) if discovery_sample_total else 0.0,
            "discovery_contrast": (discovery_contrast_total / discovery_sample_total) if discovery_sample_total else 0.0,
            "discovery_imagination": (discovery_imagination_total / discovery_sample_total) if discovery_sample_total else 0.0,
            "discovery_emotion": (discovery_emotion_total / discovery_sample_total) if discovery_sample_total else 0.0,
            "discovery_confidence": (discovery_confidence_total / discovery_sample_total) if discovery_sample_total else 0.0,
            "discovery_curiosity": (discovery_curiosity_total / discovery_sample_total) if discovery_sample_total else 0.0,
            "discovery_counter_share": (discovery_counter_share_total / discovery_batches) if discovery_batches else 0.0,
            "discovery_updates": int(discovery_config.orchestrator.total_updates) if discovery_enabled else 0,
            "transcendent_loss": (transcendent_loss_total / transcendent_sample_total) if transcendent_sample_total else 0.0,
            "transcendent_stability": (transcendent_stability_total / transcendent_sample_total) if transcendent_sample_total else 0.0,
            "transcendent_divergence": (transcendent_divergence_total / transcendent_sample_total) if transcendent_sample_total else 0.0,
            "transcendent_foresight": (transcendent_foresight_total / transcendent_sample_total) if transcendent_sample_total else 0.0,
            "transcendent_synthesis": (transcendent_synthesis_total / transcendent_sample_total) if transcendent_sample_total else 0.0,
            "transcendent_affective": (transcendent_affective_total / transcendent_sample_total) if transcendent_sample_total else 0.0,
            "transcendent_entropy": (transcendent_entropy_total / transcendent_sample_total) if transcendent_sample_total else 0.0,
            "transcendent_coherence": (transcendent_coherence_total / transcendent_batches) if transcendent_batches else 0.0,
            "transcendent_updates": int(transcendent_config.architect.total_updates) if transcendent_enabled else 0,
            "frontier_loss": (frontier_loss_total / frontier_sample_total) if frontier_sample_total else 0.0,
            "frontier_novelty": (frontier_novelty_total / frontier_sample_total) if frontier_sample_total else 0.0,
            "frontier_abstraction": (frontier_abstraction_total / frontier_sample_total) if frontier_sample_total else 0.0,
            "frontier_transfer": (frontier_transfer_total / frontier_sample_total) if frontier_sample_total else 0.0,
            "frontier_curiosity": (frontier_curiosity_total / frontier_sample_total) if frontier_sample_total else 0.0,
            "frontier_emotion": (frontier_emotion_total / frontier_sample_total) if frontier_sample_total else 0.0,
            "frontier_meta": (frontier_meta_total / frontier_sample_total) if frontier_sample_total else 0.0,
            "frontier_diversity": (frontier_diversity_total / frontier_batches) if frontier_batches else 0.0,
            "frontier_updates": int(frontier_config.catalyst.total_updates) if frontier_enabled else 0,
        },
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    return_details: bool = False,
    emotion_config: Optional[EmotionTrainingConfig] = None,
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
            emotion_features = None
            if len(batch) == 6:
                inputs, targets, _weights, attention_mask, _teacher_logits, emotion_features = batch
            elif len(batch) == 5:
                inputs, targets, _weights, attention_mask, _teacher_logits = batch
            else:
                inputs, targets, _weights, attention_mask = batch  # type: ignore[misc]
            inputs = inputs.to(device)
            targets = targets.to(device)
            attention_mask = attention_mask.to(device)
            if emotion_features is not None:
                emotion_features = emotion_features.to(device=device, dtype=torch.float32)
                if emotion_features.dim() == 1:
                    emotion_features = emotion_features.unsqueeze(0)
            supports_emotion = (
                emotion_features is not None
                and emotion_features.numel() > 0
                and emotion_config is not None
                and emotion_config.enabled
                and getattr(model, "supports_emotion_features", False)
            )

            if supports_emotion:
                logits, _, _ = model(
                    inputs,
                    attention_mask=attention_mask,
                    emotion_features=emotion_features,
                    return_components=True,
                )
            else:
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
    emotion_encoder: Optional[EmotionLexicon] = None,
    emotion_dim: int = 0,
    emotion_config: Optional[EmotionTrainingConfig] = None,
) -> Tuple[List[Tuple[str, str, float]], List[str]]:
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    confident: List[Tuple[str, str, float]] = []
    remaining: List[str] = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            ids, mask, emotion_features = _prepare_model_inputs(
                text,
                vocab=vocab,
                max_len=max_len,
                device=device,
                tokenizer=tokenizer,
                tokenizer_cache=tokenizer_cache,
                embedding_model=embedding_model,
                emotion_encoder=emotion_encoder,
                emotion_dim=emotion_dim,
            )
            supports_emotion = (
                emotion_features is not None
                and emotion_features.numel() > 0
                and getattr(model, "supports_emotion_features", False)
                and (emotion_config is None or emotion_config.enabled)
            )
            if supports_emotion:
                logits, _, _ = model(
                    ids,
                    attention_mask=mask,
                    emotion_features=emotion_features,
                    return_components=True,
                )
            else:
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


@dataclass
class ModelPrediction:
    label: str
    confidence: float
    top_predictions: List[Tuple[str, float]]


def _prepare_model_inputs(
    text: str,
    *,
    vocab: Dict[str, int],
    max_len: int,
    device: torch.device,
    tokenizer=None,
    tokenizer_cache: Optional[Callable[[str], Tuple[Sequence[int], Sequence[int]]]] = None,
    embedding_model: Optional[Callable[[str], Sequence[float]]] = None,
    emotion_encoder: Optional[EmotionLexicon] = None,
    emotion_dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    emotion_tensor: Optional[torch.Tensor] = None
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
    if emotion_encoder is not None:
        values = list(emotion_encoder.vectorise(text))
        resolved_dim = emotion_dim or len(values)
        if resolved_dim > 0:
            if len(values) < resolved_dim:
                values = values + [0.0] * (resolved_dim - len(values))
            elif len(values) > resolved_dim:
                values = values[:resolved_dim]
            emotion_tensor = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
    return ids, mask, emotion_tensor


def predict_with_trace(
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
    top_k: int = 3,
    emotion_encoder: Optional[EmotionLexicon] = None,
    emotion_dim: int = 0,
    emotion_config: Optional[EmotionTrainingConfig] = None,
) -> ModelPrediction:
    ids, mask, emotion_features = _prepare_model_inputs(
        text,
        vocab=vocab,
        max_len=max_len,
        device=device,
        tokenizer=tokenizer,
        tokenizer_cache=tokenizer_cache,
        embedding_model=embedding_model,
        emotion_encoder=emotion_encoder,
        emotion_dim=emotion_dim,
    )
    model.eval()
    with torch.no_grad():
        supports_emotion = (
            emotion_features is not None
            and emotion_features.numel() > 0
            and getattr(model, "supports_emotion_features", False)
            and (emotion_config is None or emotion_config.enabled)
        )
        if supports_emotion:
            logits, _, _ = model(
                ids,
                attention_mask=mask,
                emotion_features=emotion_features,
                return_components=True,
            )
        else:
            logits = model(ids, attention_mask=mask)
        probs = torch.softmax(logits, dim=-1)
        confidence_tensor, predicted = probs.max(dim=1)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    label = idx_to_label[predicted.item()]
    confidence = float(confidence_tensor.item())
    ranked: List[Tuple[str, float]] = []
    if top_k > 0:
        limit = min(top_k, probs.shape[-1])
        top_scores, top_indices = probs.topk(limit, dim=1)
        ranked = [
            (idx_to_label[index.item()], float(score.item()))
            for index, score in zip(top_indices[0], top_scores[0])
        ]
    return ModelPrediction(label=label, confidence=confidence, top_predictions=ranked)


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
    return_confidence: bool = False,
    emotion_encoder: Optional[EmotionLexicon] = None,
    emotion_dim: int = 0,
    emotion_config: Optional[EmotionTrainingConfig] = None,
) -> Union[str, Tuple[str, float]]:
    prediction = predict_with_trace(
        model,
        text,
        vocab=vocab,
        label_to_idx=label_to_idx,
        max_len=max_len,
        device=device,
        tokenizer=tokenizer,
        tokenizer_cache=tokenizer_cache,
        embedding_model=embedding_model,
        top_k=1 if return_confidence else 0,
        emotion_encoder=emotion_encoder,
        emotion_dim=emotion_dim,
        emotion_config=emotion_config,
    )
    if return_confidence:
        return prediction.label, prediction.confidence
    return prediction.label


@dataclass
class ResponseOutcome:
    message: str
    strategy: str
    basis: Optional[str] = None


QUESTION_STOPWORDS: Set[str] = {
    "a",
    "about",
    "am",
    "an",
    "and",
    "are",
    "be",
    "can",
    "could",
    "do",
    "does",
    "explain",
    "for",
    "give",
    "have",
    "how",
    "i",
    "is",
    "it",
    "let",
    "me",
    "please",
    "should",
    "show",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "us",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "will",
    "would",
    "you",
}


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _truncate_snippet(text: str, limit: int = 160) -> str:
    cleaned = _collapse_whitespace(text.strip())
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    truncated = cleaned[:limit]
    if " " in truncated and len(cleaned) > limit:
        truncated = truncated[: truncated.rfind(" ")]
    return truncated + "…"


def _question_focus(text: str) -> str:
    words = re.findall(r"[\w'-]+", text)
    focus_words: List[str] = []
    for word in words:
        if normalise_text(word) in QUESTION_STOPWORDS:
            continue
        focus_words.append(word)
    focus = " ".join(focus_words).strip()
    return _truncate_snippet(focus)


def _humanize_label(label: str) -> str:
    return label.replace("_", " ")


def _compose(parts: Sequence[str]) -> str:
    return " ".join(part for part in parts if part).strip()


def answer_question(text: str) -> ResponseOutcome:
    stripped = text.strip()
    cleaned = _collapse_whitespace(stripped)
    if not cleaned:
        return ResponseOutcome(
            message="I didn't catch the question—could you rephrase it so I can help?",
            strategy="question_clarification",
        )
    lowered = normalise_text(text)
    focus = _question_focus(stripped)
    basis = focus or cleaned

    def respond(message: str, strategy: str) -> ResponseOutcome:
        return ResponseOutcome(message=message, strategy=strategy, basis=basis)

    if "remind" in lowered or "reminder" in lowered:
        return respond(
            f"I'll capture the reminder about {basis} and schedule the follow-up right away.",
            "reminder_preparation",
        )
    if re.search(r"\b(when|what time)\b", lowered):
        return respond(
            f"I'll verify the timing for {basis} and get back to you with the confirmed schedule.",
            "schedule_lookup",
        )
    if re.search(r"\bwhere\b", lowered):
        return respond(
            f"I'll locate the right place for {basis} and send you the directions.",
            "location_lookup",
        )
    if re.search(r"\bwho\b", lowered):
        return respond(
            f"I'll check who is responsible for {basis} and connect you with them.",
            "ownership_lookup",
        )
    if lowered.startswith("how can i") or lowered.startswith("how to ") or "how do i" in lowered:
        action = basis
        return respond(
            f"I'll walk through the runbook for {action} and document the steps so you can follow along.",
            "process_guidance",
        )
    if "why" in lowered:
        return respond(
            f"I'll investigate why {basis} is happening and share a root-cause summary.",
            "root_cause_investigation",
        )
    if any(keyword in lowered for keyword in ["backup", "server", "database", "credential", "outage", "incident"]):
        return respond(
            f"I'll pull the operations notes for {basis} and coordinate the next steps.",
            "operations_follow_up",
        )
    if any(keyword in lowered for keyword in ["upload", "share", "send", "submit", "post"]):
        return respond(
            f"I'll confirm the correct destination for {basis} and reply with the hand-off instructions.",
            "handoff_lookup",
        )
    if any(keyword in lowered for keyword in ["issue", "problem", "broken", "crash", "error", "not working"]):
        return respond(
            f"I'll triage the issue around {basis} and keep you posted on the fix.",
            "incident_triage",
        )
    return respond(
        f"I'll research your question about {basis} and follow up with the answer shortly.",
        "general_research",
    )


def _supportive_response(label: str, text: str) -> ResponseOutcome:
    snippet = _truncate_snippet(text)
    human = _humanize_label(label)
    message = _compose(
        [
            f"I appreciate your {human} message.",
            f"I captured it as: \"{snippet}\"." if snippet else "",
            "Thanks for sharing that energy.",
        ]
    )
    return ResponseOutcome(message, "support_acknowledgement", snippet or None)


def _greeting_response(label: str, text: str) -> ResponseOutcome:
    snippet = _truncate_snippet(text)
    if label == "farewell":
        message = _compose(
            [
                "I noted your farewell message.",
                f"Captured words: \"{snippet}\"." if snippet else "",
                "Looking forward to catching up again soon.",
            ]
        )
        return ResponseOutcome(message, "farewell_acknowledgement", snippet or None)
    message = _compose(
        [
            "Great to hear from you.",
            f"I logged your greeting as: \"{snippet}\"." if snippet else "",
            "Let me know how I can assist further.",
        ]
    )
    return ResponseOutcome(message, "greeting_acknowledgement", snippet or None)


def _apology_response(label: str, text: str) -> ResponseOutcome:
    snippet = _truncate_snippet(text)
    message = _compose(
        [
            "Thanks for the apology.",
            f"Noted message: \"{snippet}\"." if snippet else "",
            "We're all squared away—let's keep moving forward.",
        ]
    )
    return ResponseOutcome(message, "apology_acknowledgement", snippet or None)


def _feedback_response(label: str, text: str) -> ResponseOutcome:
    snippet = _truncate_snippet(text)
    human = _humanize_label(label)
    message = _compose(
        [
            f"Thanks for the {human}.",
            f"I captured the details: \"{snippet}\"." if snippet else "",
            "I'll translate it into concrete improvements.",
        ]
    )
    return ResponseOutcome(message, "feedback_follow_up", snippet or None)


def _guidance_response(label: str, text: str) -> ResponseOutcome:
    snippet = _truncate_snippet(text)
    human = _humanize_label(label)
    message = _compose(
        [
            f"I appreciate the {human}.",
            f"Guidance recorded as: \"{snippet}\"." if snippet else "",
            "I'll review it and factor it into our plan.",
        ]
    )
    return ResponseOutcome(message, "guidance_review", snippet or None)


def _update_response(label: str, text: str) -> ResponseOutcome:
    snippet = _truncate_snippet(text)
    human = _humanize_label(label)
    message = _compose(
        [
            f"Thanks for the {human} update.",
            f"I archived it as: \"{snippet}\"." if snippet else "",
            "It will show up in the latest status notes.",
        ]
    )
    return ResponseOutcome(message, "update_recording", snippet or None)


def _actionable_response(label: str, text: str) -> ResponseOutcome:
    snippet = _truncate_snippet(text)
    human = _humanize_label(label)
    message = _compose(
        [
            f"I captured the {human}.",
            f"Tracked request: \"{snippet}\"." if snippet else "",
            "I'll own the follow-up and report on progress.",
        ]
    )
    strategy = "instruction_tracking" if label == "instruction" else "request_tracking"
    return ResponseOutcome(message, strategy, snippet or None)


def _reminder_response(label: str, text: str) -> ResponseOutcome:
    snippet = _truncate_snippet(text)
    message = _compose(
        [
            "I logged the reminder.",
            f"Reminder content: \"{snippet}\"." if snippet else "",
            "I'll schedule the prompt and ping you ahead of time.",
        ]
    )
    return ResponseOutcome(message, "reminder_scheduling", snippet or None)


def _fun_response(label: str, text: str) -> ResponseOutcome:
    snippet = _truncate_snippet(text)
    human = _humanize_label(label)
    message = _compose(
        [
            f"That {human} made me smile.",
            f"Highlighted moment: \"{snippet}\"." if snippet else "",
            "Thanks for brightening the workflow.",
        ]
    )
    return ResponseOutcome(message, "levity_acknowledgement", snippet or None)


def _creative_response(label: str, text: str) -> ResponseOutcome:
    snippet = _truncate_snippet(text)
    human = _humanize_label(label)
    message = _compose(
        [
            f"Your {human} sparks imagination.",
            f"I'm saving this line: \"{snippet}\"." if snippet else "",
            "I'll share it with the team for inspiration.",
        ]
    )
    return ResponseOutcome(message, "creative_acknowledgement", snippet or None)


def _generic_intent_response(label: str, text: str) -> ResponseOutcome:
    snippet = _truncate_snippet(text)
    human = _humanize_label(label)
    message = _compose(
        [
            f"I noted your {human} message.",
            f"Content: \"{snippet}\"." if snippet else "",
            "I'll keep it in context as we move forward.",
        ]
    )
    return ResponseOutcome(message, "generic_intent_response", snippet or None)


def _bind_label(handler: Callable[[str, str], ResponseOutcome], label: str) -> Callable[[str], ResponseOutcome]:
    return lambda text: handler(label, text)


SUPPORTIVE_INTENTS: Set[str] = {
    "thank_you",
    "compliment",
    "positive_statement",
    "positive_experience",
    "motivation",
}

GREETING_INTENTS: Set[str] = {"greeting", "farewell"}

ACTIONABLE_INTENTS: Set[str] = {"request", "instruction"}

GUIDANCE_INTENTS: Set[str] = {"recommendation", "advice", "suggestion"}

UPDATE_INTENTS: Set[str] = {
    "announcement",
    "observation",
    "statement",
    "fact",
    "news_headline",
    "weather_report",
    "weather_statement",
    "technical_statement",
    "technical_instruction",
    "definition",
}

FEEDBACK_INTENTS: Set[str] = {"criticism", "error_message"}

FUN_INTENTS: Set[str] = {"joke", "humor", "pun"}

CREATIVE_INTENTS: Set[str] = {
    "story_snippet",
    "poem_line",
    "quote",
    "riddle",
    "saying",
    "sarcasm",
}


RESPONSE_POLICIES: Dict[str, Callable[[str], ResponseOutcome]] = {}
for intent in SUPPORTIVE_INTENTS:
    RESPONSE_POLICIES[intent] = _bind_label(_supportive_response, intent)
for intent in GREETING_INTENTS:
    RESPONSE_POLICIES[intent] = _bind_label(_greeting_response, intent)
for intent in ACTIONABLE_INTENTS:
    RESPONSE_POLICIES[intent] = _bind_label(_actionable_response, intent)
for intent in GUIDANCE_INTENTS:
    RESPONSE_POLICIES[intent] = _bind_label(_guidance_response, intent)
for intent in UPDATE_INTENTS:
    RESPONSE_POLICIES[intent] = _bind_label(_update_response, intent)
for intent in FEEDBACK_INTENTS:
    RESPONSE_POLICIES[intent] = _bind_label(_feedback_response, intent)
for intent in FUN_INTENTS:
    RESPONSE_POLICIES[intent] = _bind_label(_fun_response, intent)
for intent in CREATIVE_INTENTS:
    RESPONSE_POLICIES[intent] = _bind_label(_creative_response, intent)

RESPONSE_POLICIES["apology"] = _bind_label(_apology_response, "apology")
RESPONSE_POLICIES["reminder"] = _bind_label(_reminder_response, "reminder")


def generate_response(label: str, text: str) -> ResponseOutcome:
    if label == "question":
        return answer_question(text)
    handler = RESPONSE_POLICIES.get(label)
    if handler is not None:
        return handler(text)
    return _generic_intent_response(label, text)


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
    synthetic_examples = int(metrics.get("synthetic_examples_added", 0))
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
        f"- **Synthetic self-play additions:** {synthetic_examples}",
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
    parser.add_argument("--self-play-rounds", type=int, default=1,
                        help="Number of synthetic self-play rounds executed after each training phase (0 disables).")
    parser.add_argument("--self-play-epochs", type=int, default=1,
                        help="Epochs of fine-tuning performed after injecting synthetic self-play examples.")
    parser.add_argument("--self-play-per-label", type=int, default=3,
                        help="Target number of accepted self-play examples per label in each round.")
    parser.add_argument("--self-play-max-length", type=int, default=32,
                        help="Maximum token length for generated self-play prompts.")
    parser.add_argument("--self-play-samples", type=int, default=4,
                        help="Monte Carlo dropout samples used to self-evaluate synthetic prompts.")
    parser.add_argument("--self-play-min-confidence", type=float, default=0.6,
                        help="Minimum blended confidence required to accept a synthetic self-play example.")
    parser.add_argument("--self-play-consistency", type=float, default=0.65,
                        help="Minimum agreement ratio between stochastic predictions when vetting self-play examples.")
    parser.add_argument("--adaptive-curriculum", action="store_true",
                        help="Enable difficulty-aware curriculum weighting for the supervised corpus.")
    parser.add_argument("--curriculum-start-epoch", type=int, default=2,
                        help="Global epoch after which curriculum weighting begins to adjust samples.")
    parser.add_argument("--curriculum-momentum", type=float, default=0.65,
                        help="Smoothing factor applied when blending old and new curriculum weights (0 disables smoothing).")
    parser.add_argument("--curriculum-min-multiplier", type=float, default=0.5,
                        help="Lower bound multiplier applied to each sample's base weight when curriculum is active.")
    parser.add_argument("--curriculum-max-multiplier", type=float, default=3.5,
                        help="Upper bound multiplier applied to each sample's base weight when curriculum is active.")
    parser.add_argument("--curriculum-hard-boost", type=float, default=2.2,
                        help="Strength of the boost applied to harder samples (higher values emphasise difficult examples).")
    parser.add_argument("--curriculum-difficulty-power", type=float, default=1.4,
                        help="Exponent applied to difficulty when computing curriculum boosts (>=0.5).")
    parser.add_argument("--enable-emotion-reasoner", action="store_true",
                        help="Enable affect-aware reasoning that fuses emotion prototypes into the classifier.")
    parser.add_argument("--emotion-consistency-weight", type=float, default=0.35,
                        help="Weight applied to the emotion alignment auxiliary loss when the reasoner is enabled.")
    parser.add_argument("--emotion-expectation-temperature", type=float, default=1.0,
                        help="Temperature applied to logits before deriving expected emotion vectors (>=0).")
    parser.add_argument("--emotion-prototype-smoothing", type=float, default=0.05,
                        help="Additive smoothing used when maintaining emotion prototypes per intent label.")
    parser.add_argument("--emotion-fusion-dropout", type=float, default=0.15,
                        help="Dropout probability inside the emotion fusion adapter (0 disables).")
    parser.add_argument("--meta-introspector", action="store_true",
                        help="Enable the meta-cognitive introspector that learns class prototypes and curiosity-driven losses.")
    parser.add_argument("--meta-attraction-weight", type=float, default=0.15,
                        help="Strength of the attraction loss that pulls representations toward their class prototype.")
    parser.add_argument("--meta-repulsion-weight", type=float, default=0.08,
                        help="Strength of the repulsion margin that separates representations from other prototypes.")
    parser.add_argument("--meta-discovery-weight", type=float, default=0.05,
                        help="Strength of the discovery term that penalises shallow confidence gaps during introspection.")
    parser.add_argument("--meta-margin", type=float, default=1.5,
                        help="Margin radius used when measuring prototype repulsion distance.")
    parser.add_argument("--meta-min-confidence-gap", type=float, default=0.25,
                        help="Target minimum confidence gap maintained by the introspector's discovery pressure.")
    parser.add_argument("--meta-momentum", type=float, default=0.15,
                        help="Momentum applied when updating class prototypes and curiosity statistics.")
    parser.add_argument("--meta-history", type=int, default=64,
                        help="Number of high-curiosity traces retained for diagnostics in the introspection ledger.")
    parser.add_argument("--meta-temperature", type=float, default=1.0,
                        help="Global temperature scaling applied to the combined meta-cognitive regulariser.")
    parser.add_argument("--neuro-symbolic-reasoner", action="store_true",
                        help="Enable the neuro-symbolic reasoner that fuses lexical concept graphs with learned representations.")
    parser.add_argument("--neuro-structural-weight", type=float, default=0.35,
                        help="Weight applied to the neuro-symbolic structural alignment penalty.")
    parser.add_argument("--neuro-semantic-weight", type=float, default=0.25,
                        help="Weight applied to feature-level semantic alignment inside the neuro-symbolic reasoner.")
    parser.add_argument("--neuro-affective-weight", type=float, default=0.12,
                        help="Weight applied to affective/emotion consistency inside the neuro-symbolic reasoner.")
    parser.add_argument("--neuro-temperature", type=float, default=1.0,
                        help="Temperature used when projecting logits into the neuro-symbolic neighbourhood distribution.")
    parser.add_argument("--neuro-self-loop", type=float, default=0.6,
                        help="Proportion of probability mass retained on the true label during neuro-symbolic smoothing (0-1).")
    parser.add_argument("--neuro-lexical-weight", type=float, default=0.55,
                        help="Blend factor between lexical priors and dynamic graph structure in the neuro-symbolic reasoner (0-1).")
    parser.add_argument("--neuro-graph-momentum", type=float, default=0.25,
                        help="Momentum applied when updating the neuro-symbolic concept graph (0-1).")
    parser.add_argument("--neuro-feature-momentum", type=float, default=0.15,
                        help="Momentum applied when refreshing neuro-symbolic prototypes from latent features (0-1).")
    parser.add_argument("--neuro-min-confidence", type=float, default=0.35,
                        help="Minimum confidence required for a sample to influence the neuro-symbolic graph (0-1).")
    parser.add_argument("--neuro-history", type=int, default=96,
                        help="Maximum number of neuro-symbolic diagnostic entries retained for metadata summaries.")
    parser.add_argument("--neuro-max-keywords", type=int, default=18,
                        help="Maximum lexical keywords stored per intent for the neuro-symbolic concept summaries.")
    parser.add_argument("--self-discovery", action="store_true",
                        help="Enable the self-discovery orchestrator that tracks counterfactual prototypes and imagination cues.")
    parser.add_argument("--discovery-alignment-weight", type=float, default=0.32,
                        help="Weight applied to the alignment term that pulls features toward reflective prototypes.")
    parser.add_argument("--discovery-contrast-weight", type=float, default=0.26,
                        help="Weight applied to the counterfactual contrast penalty inside the self-discovery module.")
    parser.add_argument("--discovery-imagination-weight", type=float, default=0.18,
                        help="Weight applied to the imagination term that compares latent predictions against expected manifolds.")
    parser.add_argument("--discovery-emotion-weight", type=float, default=0.1,
                        help="Weight applied to affective alignment within the self-discovery orchestrator when emotion cues exist.")
    parser.add_argument("--discovery-temperature", type=float, default=1.0,
                        help="Temperature used when shaping discovery logits before computing auxiliary expectations.")
    parser.add_argument("--discovery-min-confidence", type=float, default=0.45,
                        help="Confidence threshold below which samples trigger counterfactual imagination updates (0-1).")
    parser.add_argument("--discovery-margin", type=float, default=0.15,
                        help="Margin applied to cosine contrast when discouraging counterfactual collapse.")
    parser.add_argument("--discovery-feature-momentum", type=float, default=0.24,
                        help="Momentum applied when refreshing positive discovery prototypes from latent features (0-1).")
    parser.add_argument("--discovery-counter-momentum", type=float, default=0.3,
                        help="Momentum applied when updating counterfactual prototypes for difficult examples (0-1).")
    parser.add_argument("--discovery-imagination-momentum", type=float, default=0.18,
                        help="Momentum applied when integrating expectation traces within the self-discovery orchestrator (0-1).")
    parser.add_argument("--discovery-curiosity-weight", type=float, default=0.5,
                        help="Scalar multiplying the curiosity term that rewards confident coverage inside self-discovery.")
    parser.add_argument("--discovery-history", type=int, default=128,
                        help="Number of diagnostic events preserved for the self-discovery orchestrator.")
    parser.add_argument("--transcendent-cognition", action="store_true",
                        help="Enable the transcendent cognition architect that fuses foresight, synthesis, and affect loops.")
    parser.add_argument("--transcendent-stability-weight", type=float, default=0.85,
                        help="Weight applied to the stability component of the transcendent cognition regulariser.")
    parser.add_argument("--transcendent-divergence-weight", type=float, default=0.55,
                        help="Weight applied to divergence pressure against rival intent trajectories.")
    parser.add_argument("--transcendent-foresight-weight", type=float, default=0.5,
                        help="Weight applied to the foresight alignment term for imagined futures.")
    parser.add_argument("--transcendent-synthesis-weight", type=float, default=0.45,
                        help="Weight applied to synthesis coherence across predicted transitions.")
    parser.add_argument("--transcendent-affective-weight", type=float, default=0.35,
                        help="Weight applied to affective alignment when emotion vectors are available.")
    parser.add_argument("--transcendent-entropy-weight", type=float, default=0.25,
                        help="Weight applied to entropy modulation inside the transcendent cognition loss.")
    parser.add_argument("--transcendent-temperature", type=float, default=1.0,
                        help="Temperature applied before constructing transcendent cognition distributions.")
    parser.add_argument("--transcendent-margin", type=float, default=0.75,
                        help="Margin applied when measuring divergence in transcendent cognition.")
    parser.add_argument("--transcendent-feature-momentum", type=float, default=0.18,
                        help="Momentum used for updating class anchors inside the transcendent architect.")
    parser.add_argument("--transcendent-counter-momentum", type=float, default=0.24,
                        help="Momentum used when updating counterfactual bridges inside transcendent cognition.")
    parser.add_argument("--transcendent-transition-momentum", type=float, default=0.12,
                        help="Momentum used when updating transition priors inside transcendent cognition.")
    parser.add_argument("--transcendent-imagination-momentum", type=float, default=0.16,
                        help="Momentum used for imagination traces inside the transcendent architect.")
    parser.add_argument("--transcendent-history", type=int, default=160,
                        help="Number of diagnostic entries retained by the transcendent cognition architect.")
    parser.add_argument("--transcendent-max-glimpses", type=int, default=4,
                        help="Number of top hypotheses retained when trimming transcendent cognition distributions (>=1).")
    parser.add_argument("--frontier-intelligence", action="store_true",
                        help="Enable the frontier intelligence catalyst that scouts novel abstractions and transfer paths.")
    parser.add_argument("--frontier-novelty-weight", type=float, default=0.6,
                        help="Weight applied to the novelty scouting term in the frontier intelligence loss.")
    parser.add_argument("--frontier-abstraction-weight", type=float, default=0.5,
                        help="Weight applied to abstraction alignment inside the frontier intelligence loss.")
    parser.add_argument("--frontier-transfer-weight", type=float, default=0.55,
                        help="Weight applied to bridge-based transfer inside the frontier intelligence loss.")
    parser.add_argument("--frontier-curiosity-weight", type=float, default=0.4,
                        help="Weight applied to the curiosity activation term inside the frontier intelligence loss.")
    parser.add_argument("--frontier-emotion-weight", type=float, default=0.35,
                        help="Weight applied to emotion expectation alignment for the frontier catalyst.")
    parser.add_argument("--frontier-meta-weight", type=float, default=0.25,
                        help="Weight applied to meta-bias regularisation for frontier intelligence.")
    parser.add_argument("--frontier-temperature", type=float, default=1.0,
                        help="Temperature applied before constructing frontier intelligence distributions.")
    parser.add_argument("--frontier-margin", type=float, default=0.65,
                        help="Margin target used when encouraging curiosity and novelty in frontier intelligence.")
    parser.add_argument("--frontier-concept-momentum", type=float, default=0.22,
                        help="Momentum used when updating frontier intelligence concept prototypes.")
    parser.add_argument("--frontier-bridge-momentum", type=float, default=0.18,
                        help="Momentum used when updating frontier transfer bridges.")
    parser.add_argument("--frontier-novelty-momentum", type=float, default=0.3,
                        help="Momentum applied to novelty exemplars in the frontier catalyst.")
    parser.add_argument("--frontier-meta-momentum", type=float, default=0.12,
                        help="Momentum applied to the meta-bias trace within frontier intelligence.")
    parser.add_argument("--frontier-emotion-momentum", type=float, default=0.2,
                        help="Momentum applied when updating emotion prototypes inside frontier intelligence.")
    parser.add_argument("--frontier-history", type=int, default=192,
                        help="Number of history entries retained by the frontier intelligence catalyst (>=1).")
    parser.add_argument("--self-play-weight", type=float, default=0.35,
                        help="Base loss weight assigned to accepted self-play examples.")
    parser.add_argument("--self-play-confidence-power", type=float, default=1.2,
                        help="Exponent applied to the confidence ratio when scaling self-play weights.")
    parser.add_argument("--self-play-max-weight-multiplier", type=float, default=2.5,
                        help="Maximum multiple of --self-play-weight permitted for synthetic examples.")
    parser.add_argument("--self-play-temperature", type=float, default=1.0,
                        help="Sampling temperature applied to the label n-gram generators (values <1 sharpen, >1 smooth).")
    parser.add_argument("--self-play-ngram-order", type=int, default=3,
                        help="Order of the label-specific n-gram model used to draft synthetic prompts.")
    parser.add_argument("--self-play-require-match", dest="self_play_require_match", action="store_true",
                        help="Only keep synthetic examples when the predicted label matches the generator label.")
    parser.add_argument("--self-play-allow-mismatch", dest="self_play_require_match", action="store_false",
                        help="Allow synthetic examples even when the predicted label differs from the generator label.")
    parser.set_defaults(self_play_require_match=True)
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
    args, unknown_args = parser.parse_known_args()
    unknown_args, ignored_kernel_args = _strip_ipykernel_arguments(unknown_args)
    if ignored_kernel_args:
        print(
            "Ignoring IPython kernel launcher arguments:",
            " ".join(ignored_kernel_args),
        )
    if unknown_args:
        parser.error("unrecognized arguments: " + " ".join(unknown_args))

    args.dataset = resolve_training_input_path(
        args.dataset,
        description="labelled dataset",
        flag="--dataset",
        search_names=("intent_dataset.csv",),
    )
    if args.unlabeled_dataset is not None:
        args.unlabeled_dataset = resolve_training_input_path(
            args.unlabeled_dataset,
            description="unlabeled dataset",
            flag="--unlabeled-dataset",
        )

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
    if args.self_play_rounds < 0:
        parser.error("--self-play-rounds must be non-negative.")
    if args.self_play_epochs < 0:
        parser.error("--self-play-epochs must be non-negative.")
    if args.self_play_per_label < 0:
        parser.error("--self-play-per-label must be non-negative.")
    if args.self_play_max_length <= 0:
        parser.error("--self-play-max-length must be positive.")
    if args.self_play_samples < 1:
        parser.error("--self-play-samples must be at least 1.")
    if not 0 <= args.self_play_min_confidence <= 1:
        parser.error("--self-play-min-confidence must lie in [0, 1].")
    if not 0 <= args.self_play_consistency <= 1:
        parser.error("--self-play-consistency must lie in [0, 1].")
    if args.self_play_weight <= 0:
        parser.error("--self-play-weight must be positive.")
    if args.self_play_confidence_power < 0:
        parser.error("--self-play-confidence-power must be non-negative.")
    if args.self_play_max_weight_multiplier < 1:
        parser.error("--self-play-max-weight-multiplier must be at least 1.")
    if args.self_play_temperature <= 0:
        parser.error("--self-play-temperature must be positive.")
    if args.self_play_ngram_order < 1:
        parser.error("--self-play-ngram-order must be at least 1.")
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
    if args.curriculum_min_multiplier <= 0:
        parser.error("--curriculum-min-multiplier must be positive.")
    if args.curriculum_max_multiplier < args.curriculum_min_multiplier:
        parser.error("--curriculum-max-multiplier must be >= --curriculum-min-multiplier.")
    if not 0 <= args.curriculum_momentum < 1:
        parser.error("--curriculum-momentum must lie in [0, 1).")
    if args.curriculum_hard_boost < 0:
        parser.error("--curriculum-hard-boost must be non-negative.")
    if args.curriculum_difficulty_power < 0.5:
        parser.error("--curriculum-difficulty-power must be at least 0.5.")
    if args.curriculum_start_epoch < 0:
        parser.error("--curriculum-start-epoch must be non-negative.")
    if args.emotion_consistency_weight < 0:
        parser.error("--emotion-consistency-weight must be non-negative.")
    if args.emotion_expectation_temperature <= 0:
        parser.error("--emotion-expectation-temperature must be strictly positive.")
    if args.emotion_prototype_smoothing < 0:
        parser.error("--emotion-prototype-smoothing must be non-negative.")
    if not 0 <= args.emotion_fusion_dropout < 1:
        parser.error("--emotion-fusion-dropout must lie in [0, 1).")
    if args.meta_attraction_weight < 0:
        parser.error("--meta-attraction-weight must be non-negative.")
    if args.meta_repulsion_weight < 0:
        parser.error("--meta-repulsion-weight must be non-negative.")
    if args.meta_discovery_weight < 0:
        parser.error("--meta-discovery-weight must be non-negative.")
    if args.meta_margin <= 0:
        parser.error("--meta-margin must be strictly positive.")
    if args.meta_min_confidence_gap < 0:
        parser.error("--meta-min-confidence-gap must be non-negative.")
    if not 0 <= args.meta_momentum < 1:
        parser.error("--meta-momentum must lie in [0, 1).")
    if args.meta_history < 1:
        parser.error("--meta-history must be at least 1.")
    if args.meta_temperature <= 0:
        parser.error("--meta-temperature must be strictly positive.")
    if args.neuro_structural_weight < 0:
        parser.error("--neuro-structural-weight must be non-negative.")
    if args.neuro_semantic_weight < 0:
        parser.error("--neuro-semantic-weight must be non-negative.")
    if args.neuro_affective_weight < 0:
        parser.error("--neuro-affective-weight must be non-negative.")
    if args.neuro_temperature <= 0:
        parser.error("--neuro-temperature must be strictly positive.")
    if not 0 <= args.neuro_self_loop <= 1:
        parser.error("--neuro-self-loop must lie in [0, 1].")
    if not 0 <= args.neuro_lexical_weight <= 1:
        parser.error("--neuro-lexical-weight must lie in [0, 1].")
    if not 0 <= args.neuro_graph_momentum < 1:
        parser.error("--neuro-graph-momentum must lie in [0, 1).")
    if not 0 <= args.neuro_feature_momentum < 1:
        parser.error("--neuro-feature-momentum must lie in [0, 1).")
    if not 0 <= args.neuro_min_confidence <= 1:
        parser.error("--neuro-min-confidence must lie in [0, 1].")
    if args.neuro_history < 1:
        parser.error("--neuro-history must be at least 1.")
    if args.neuro_max_keywords < 1:
        parser.error("--neuro-max-keywords must be at least 1.")
    if args.discovery_alignment_weight < 0:
        parser.error("--discovery-alignment-weight must be non-negative.")
    if args.discovery_contrast_weight < 0:
        parser.error("--discovery-contrast-weight must be non-negative.")
    if args.discovery_imagination_weight < 0:
        parser.error("--discovery-imagination-weight must be non-negative.")
    if args.discovery_emotion_weight < 0:
        parser.error("--discovery-emotion-weight must be non-negative.")
    if args.discovery_temperature <= 0:
        parser.error("--discovery-temperature must be strictly positive.")
    if not 0 <= args.discovery_min_confidence <= 1:
        parser.error("--discovery-min-confidence must lie in [0, 1].")
    if args.discovery_margin < 0:
        parser.error("--discovery-margin must be non-negative.")
    if not 0 <= args.discovery_feature_momentum < 1:
        parser.error("--discovery-feature-momentum must lie in [0, 1).")
    if not 0 <= args.discovery_counter_momentum < 1:
        parser.error("--discovery-counter-momentum must lie in [0, 1).")
    if not 0 <= args.discovery_imagination_momentum < 1:
        parser.error("--discovery-imagination-momentum must lie in [0, 1).")
    if args.discovery_curiosity_weight < 0:
        parser.error("--discovery-curiosity-weight must be non-negative.")
    if args.discovery_history < 1:
        parser.error("--discovery-history must be at least 1.")
    if args.transcendent_stability_weight < 0:
        parser.error("--transcendent-stability-weight must be non-negative.")
    if args.transcendent_divergence_weight < 0:
        parser.error("--transcendent-divergence-weight must be non-negative.")
    if args.transcendent_foresight_weight < 0:
        parser.error("--transcendent-foresight-weight must be non-negative.")
    if args.transcendent_synthesis_weight < 0:
        parser.error("--transcendent-synthesis-weight must be non-negative.")
    if args.transcendent_affective_weight < 0:
        parser.error("--transcendent-affective-weight must be non-negative.")
    if args.transcendent_entropy_weight < 0:
        parser.error("--transcendent-entropy-weight must be non-negative.")
    if args.transcendent_temperature <= 0:
        parser.error("--transcendent-temperature must be strictly positive.")
    if args.transcendent_margin < 0:
        parser.error("--transcendent-margin must be non-negative.")
    if not 0 <= args.transcendent_feature_momentum < 1:
        parser.error("--transcendent-feature-momentum must lie in [0, 1).")
    if not 0 <= args.transcendent_counter_momentum < 1:
        parser.error("--transcendent-counter-momentum must lie in [0, 1).")
    if not 0 <= args.transcendent_transition_momentum < 1:
        parser.error("--transcendent-transition-momentum must lie in [0, 1).")
    if not 0 <= args.transcendent_imagination_momentum < 1:
        parser.error("--transcendent-imagination-momentum must lie in [0, 1).")
    if args.transcendent_history < 1:
        parser.error("--transcendent-history must be at least 1.")
    if args.transcendent_max_glimpses < 1:
        parser.error("--transcendent-max-glimpses must be at least 1.")
    if args.frontier_novelty_weight < 0:
        parser.error("--frontier-novelty-weight must be non-negative.")
    if args.frontier_abstraction_weight < 0:
        parser.error("--frontier-abstraction-weight must be non-negative.")
    if args.frontier_transfer_weight < 0:
        parser.error("--frontier-transfer-weight must be non-negative.")
    if args.frontier_curiosity_weight < 0:
        parser.error("--frontier-curiosity-weight must be non-negative.")
    if args.frontier_emotion_weight < 0:
        parser.error("--frontier-emotion-weight must be non-negative.")
    if args.frontier_meta_weight < 0:
        parser.error("--frontier-meta-weight must be non-negative.")
    if args.frontier_temperature <= 0:
        parser.error("--frontier-temperature must be strictly positive.")
    if args.frontier_margin < 0:
        parser.error("--frontier-margin must be non-negative.")
    if not 0 <= args.frontier_concept_momentum < 1:
        parser.error("--frontier-concept-momentum must lie in [0, 1).")
    if not 0 <= args.frontier_bridge_momentum < 1:
        parser.error("--frontier-bridge-momentum must lie in [0, 1).")
    if not 0 <= args.frontier_novelty_momentum < 1:
        parser.error("--frontier-novelty-momentum must lie in [0, 1).")
    if not 0 <= args.frontier_meta_momentum < 1:
        parser.error("--frontier-meta-momentum must lie in [0, 1).")
    if not 0 <= args.frontier_emotion_momentum < 1:
        parser.error("--frontier-emotion-momentum must lie in [0, 1).")
    if args.frontier_history < 1:
        parser.error("--frontier-history must be at least 1.")

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
    idx_to_label_list = ["?"] * num_classes
    for label, idx in label_to_idx.items():
        idx_to_label_list[idx] = label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emotion_enabled = bool(args.enable_emotion_reasoner)
    emotion_lexicon = EmotionLexicon() if emotion_enabled else None
    emotion_dim = len(emotion_lexicon.emotions) if emotion_lexicon is not None else 0

    def build_model(target_device: Optional[torch.device] = None) -> nn.Module:
        destination = target_device or device
        if args.encoder_type == "transformer":
            model_obj = TransformerIntentModel(args.transformer_model, num_classes)
            if tokenizer_obj is not None and hasattr(model_obj, "model") and hasattr(model_obj.model, "resize_token_embeddings"):
                model_obj.model.resize_token_embeddings(len(tokenizer_obj))
            if emotion_enabled and emotion_dim > 0:
                model_obj = EmotionallyAdaptiveModel(
                    model_obj,
                    num_classes=num_classes,
                    num_emotions=emotion_dim,
                    dropout=args.emotion_fusion_dropout,
                )
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
            if emotion_enabled and emotion_dim > 0:
                model_obj = EmotionallyAdaptiveModel(
                    model_obj,
                    num_classes=num_classes,
                    num_emotions=emotion_dim,
                    dropout=args.emotion_fusion_dropout,
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
        if emotion_enabled and emotion_dim > 0:
            model_obj = EmotionallyAdaptiveModel(
                model_obj,
                num_classes=num_classes,
                num_emotions=emotion_dim,
                dropout=args.emotion_fusion_dropout,
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
    evaluation_inputs = [
        "Can you remind me to restart the backup server at 9 pm tonight?",
        "What time are we presenting the quarterly results to the board?",
        "Where should I upload the signed vendor contracts?",
        "The mobile app crashes whenever I tap the export report button.",
        "How do I reset the staging database credentials?",
        "Thanks for staying late to help with the database migration!",
        "Please archive the outdated campaign assets before Thursday.",
        "Why isn't the conference call link working for the European team?",
        "Our guests loved the espresso bar during the product launch party.",
        "Who is leading the keynote rehearsal tomorrow afternoon?",
        "I can't believe how quickly the support crew resolved that outage.",
    ]
    if args.encoder_type == "transformer" and tokenizer_cache_fn is not None:
        populate_tokenizer_cache(evaluation_inputs, "evaluation showcase set")

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

        train_emotion_vectors: List[List[float]] = []
        fold_emotion_memory: Optional[EmotionPrototypeMemory] = None
        fold_emotion_config: Optional[EmotionTrainingConfig] = None
        if emotion_enabled and emotion_lexicon is not None and emotion_dim > 0:
            train_emotion_vectors = [list(emotion_lexicon.vectorise(text)) for text in train_texts]
            fold_emotion_memory = EmotionPrototypeMemory(
                num_classes,
                emotion_dim,
                smoothing=args.emotion_prototype_smoothing,
            )
            fold_emotion_memory.register_vectors(
                [label_to_idx[label] for label in train_labels],
                train_emotion_vectors,
                weights=train_weights,
            )

        fold_meta_config: Optional[MetaCognitiveConfig] = None
        if args.meta_introspector:
            fold_meta_config = MetaCognitiveConfig(
                introspector=MetaCognitiveIntrospector(
                    num_classes,
                    momentum=args.meta_momentum,
                    margin=args.meta_margin,
                    history_limit=args.meta_history,
                ),
                enabled=True,
                attraction_weight=args.meta_attraction_weight,
                repulsion_weight=args.meta_repulsion_weight,
                discovery_weight=args.meta_discovery_weight,
                gap_margin=args.meta_min_confidence_gap,
                temperature=args.meta_temperature,
            )

        fold_neuro_config: Optional[NeuroSymbolicConfig] = None
        if args.neuro_symbolic_reasoner:
            lexical_profiles, lexical_keywords = build_label_concept_library(
                train_texts,
                train_labels,
                label_to_idx=label_to_idx,
                max_keywords=args.neuro_max_keywords,
            )
            fold_neuro_config = NeuroSymbolicConfig(
                reasoner=NeuroSymbolicReasoner(
                    num_classes,
                    idx_to_label=idx_to_label_list,
                    lexical_profiles=lexical_profiles,
                    lexical_keywords=lexical_keywords,
                    lexical_weight=args.neuro_lexical_weight,
                    graph_momentum=args.neuro_graph_momentum,
                    feature_momentum=args.neuro_feature_momentum,
                    min_confidence=args.neuro_min_confidence,
                    history_limit=args.neuro_history,
                    emotion_dim=emotion_dim if emotion_enabled else 0,
                ),
                enabled=True,
                structural_weight=args.neuro_structural_weight,
                semantic_weight=args.neuro_semantic_weight,
                affective_weight=args.neuro_affective_weight,
                temperature=args.neuro_temperature,
                self_loop=args.neuro_self_loop,
            )

        fold_discovery_config: Optional[SelfDiscoveryConfig] = None
        if args.self_discovery:
            fold_discovery_config = SelfDiscoveryConfig(
                orchestrator=SelfDiscoveryOrchestrator(
                    num_classes,
                    feature_momentum=args.discovery_feature_momentum,
                    counter_momentum=args.discovery_counter_momentum,
                    imagination_momentum=args.discovery_imagination_momentum,
                    curiosity_weight=args.discovery_curiosity_weight,
                    history_limit=args.discovery_history,
                ),
                enabled=True,
                alignment_weight=args.discovery_alignment_weight,
                contrast_weight=args.discovery_contrast_weight,
                imagination_weight=args.discovery_imagination_weight,
                emotion_weight=args.discovery_emotion_weight,
                temperature=args.discovery_temperature,
                min_confidence=args.discovery_min_confidence,
                margin=args.discovery_margin,
            )

        fold_transcendent_config: Optional[TranscendentCognitionConfig] = None
        if args.transcendent_cognition:
            fold_transcendent_config = TranscendentCognitionConfig(
                architect=TranscendentCognitionEngine(
                    num_classes,
                    feature_momentum=args.transcendent_feature_momentum,
                    counter_momentum=args.transcendent_counter_momentum,
                    transition_momentum=args.transcendent_transition_momentum,
                    imagination_momentum=args.transcendent_imagination_momentum,
                    history_limit=args.transcendent_history,
                    max_glimpses=args.transcendent_max_glimpses,
                ),
                enabled=True,
                stability_weight=args.transcendent_stability_weight,
                divergence_weight=args.transcendent_divergence_weight,
                foresight_weight=args.transcendent_foresight_weight,
                synthesis_weight=args.transcendent_synthesis_weight,
                affective_weight=args.transcendent_affective_weight,
                entropy_weight=args.transcendent_entropy_weight,
                temperature=args.transcendent_temperature,
                margin=args.transcendent_margin,
            )

        fold_frontier_config: Optional[FrontierIntelligenceConfig] = None
        if args.frontier_intelligence:
            fold_frontier_config = FrontierIntelligenceConfig(
                catalyst=FrontierIntelligenceEngine(
                    num_classes,
                    concept_momentum=args.frontier_concept_momentum,
                    bridge_momentum=args.frontier_bridge_momentum,
                    novelty_momentum=args.frontier_novelty_momentum,
                    meta_momentum=args.frontier_meta_momentum,
                    emotion_momentum=args.frontier_emotion_momentum,
                    history_limit=args.frontier_history,
                ),
                enabled=True,
                novelty_weight=args.frontier_novelty_weight,
                abstraction_weight=args.frontier_abstraction_weight,
                transfer_weight=args.frontier_transfer_weight,
                curiosity_weight=args.frontier_curiosity_weight,
                emotion_weight=args.frontier_emotion_weight,
                meta_weight=args.frontier_meta_weight,
                temperature=args.frontier_temperature,
                margin=args.frontier_margin,
            )

        curriculum_manager = None
        if args.adaptive_curriculum:
            curriculum_manager = AdaptiveCurriculum(
                start_epoch=args.curriculum_start_epoch,
                momentum=args.curriculum_momentum,
                min_multiplier=args.curriculum_min_multiplier,
                max_multiplier=args.curriculum_max_multiplier,
                hard_boost=args.curriculum_hard_boost,
                difficulty_power=args.curriculum_difficulty_power,
            )
            curriculum_manager.register_samples(train_texts, train_weights)

        val_texts = [texts[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        if not val_texts:
            raise RuntimeError("Validation split produced no examples; adjust --test-ratio or --folds.")

        val_emotion_vectors: Optional[List[List[float]]] = None
        if emotion_enabled and emotion_lexicon is not None and emotion_dim > 0:
            val_emotion_vectors = [list(emotion_lexicon.vectorise(text)) for text in val_texts]

        val_dataset = IntentDataset(
            val_texts,
            val_labels,
            vocab=vocab,
            label_to_idx=label_to_idx,
            max_len=max_seq_len,
            tokenizer=tokenizer_obj,
            tokenizer_cache=tokenizer_cache_fn,
            embedding_model=embedding_fn,
            emotion_vectors=val_emotion_vectors,
            emotion_dim=emotion_dim if emotion_enabled else 0,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        if fold_emotion_memory is not None:
            fold_emotion_config = EmotionTrainingConfig(
                memory=fold_emotion_memory,
                weight=args.emotion_consistency_weight,
                temperature=args.emotion_expectation_temperature,
                enabled=emotion_enabled,
            )

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
        self_play_rounds: List[Dict[str, object]] = []
        augmentation_events: List[Dict[str, object]] = []
        pseudo_examples_store: List[Tuple[str, str, float]] = []
        self_play_examples_store: List[Tuple[str, str, float]] = []
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
        total_self_play_added = 0
        total_augmented_examples = 0
        self_play_round_counter = 0

        grad_acc_steps = max(1, args.grad_accumulation_steps)
        augmentation_rng = random.Random(args.seed * 1009 + fold_index)
        self_play_rng = random.Random(args.seed * 4243 + fold_index)
        existing_texts: Set[str] = set(train_texts)

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
            nonlocal total_augmented_examples, swa_scheduler_obj, train_emotion_vectors, fold_emotion_memory, fold_emotion_config
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
            if emotion_enabled and emotion_lexicon is not None and emotion_dim > 0:
                augmented_emotion_vectors = list(train_emotion_vectors)
                base_len = len(train_emotion_vectors)
                if len(augmented_texts) > base_len:
                    for new_text in augmented_texts[base_len:]:
                        augmented_emotion_vectors.append(list(emotion_lexicon.vectorise(new_text)))
            else:
                augmented_emotion_vectors = None

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
                emotion_vectors=augmented_emotion_vectors,
                emotion_dim=emotion_dim if emotion_enabled else 0,
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
                    emotion_config=fold_emotion_config,
                    meta_config=fold_meta_config,
                    neuro_config=fold_neuro_config,
                    discovery_config=fold_discovery_config,
                    transcendent_config=fold_transcendent_config,
                    frontier_config=fold_frontier_config,
                )

                optimizer_step_counter += stats["optimizer_steps"]
                ema_update_counter += stats["ema_updates"]
                swa_update_counter += stats["swa_updates"]

                curriculum_summary: Optional[Dict[str, object]] = None
                if (
                    curriculum_manager is not None
                    and global_epoch >= args.curriculum_start_epoch
                ):
                    curriculum_dataset = IntentDataset(
                        train_texts,
                        train_labels,
                        vocab=vocab,
                        label_to_idx=label_to_idx,
                        max_len=max_seq_len,
                        sample_weights=train_weights,
                        tokenizer=tokenizer_obj,
                        tokenizer_cache=tokenizer_cache_fn,
                        embedding_model=embedding_fn,
                        emotion_vectors=train_emotion_vectors if (emotion_enabled and emotion_dim > 0) else None,
                        emotion_dim=emotion_dim if emotion_enabled else 0,
                    )
                    curriculum_loader = DataLoader(
                        curriculum_dataset,
                        batch_size=args.batch_size,
                    )
                    _, _, train_targets_detail, _train_predictions_detail, train_probabilities = evaluate(
                        model,
                        curriculum_loader,
                        criterion,
                        device,
                        return_details=True,
                        emotion_config=fold_emotion_config,
                    )
                    curriculum_summary = curriculum_manager.update_difficulties(
                        epoch=global_epoch,
                        stage=stage_name,
                        texts=train_texts,
                        labels=train_labels,
                        weights=train_weights,
                        targets=train_targets_detail,
                        probabilities=train_probabilities,
                        idx_to_label=idx_to_label_list,
                        snippet_fn=_truncate_snippet,
                    )
                    curriculum_manager.apply(train_texts, train_weights)
                    base_limit = min(len(train_texts), len(train_dataset.examples))
                    for idx in range(base_limit):
                        train_dataset.examples[idx].weight = float(train_weights[idx])

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

                val_loss, val_acc = evaluate(eval_model, val_loader, criterion, device, emotion_config=fold_emotion_config)
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
                        "emotion_alignment": float(stats.get("emotion_alignment", 0.0)),
                        "meta_loss": float(stats.get("meta_loss", 0.0)),
                        "meta_attraction": float(stats.get("meta_attraction", 0.0)),
                        "meta_repulsion": float(stats.get("meta_repulsion", 0.0)),
                        "meta_novelty": float(stats.get("meta_novelty", 0.0)),
                        "meta_gap": float(stats.get("meta_gap", 0.0)),
                        "meta_entropy": float(stats.get("meta_entropy", 0.0)),
                        "meta_coverage": float(stats.get("meta_coverage", 0.0)),
                        "meta_updates": float(stats.get("meta_updates", 0.0)),
                        "neuro_loss": float(stats.get("neuro_loss", 0.0)),
                        "neuro_structural": float(stats.get("neuro_structural", 0.0)),
                        "neuro_semantic": float(stats.get("neuro_semantic", 0.0)),
                        "neuro_affective": float(stats.get("neuro_affective", 0.0)),
                        "neuro_entropy": float(stats.get("neuro_entropy", 0.0)),
                        "neuro_cohesion": float(stats.get("neuro_cohesion", 0.0)),
                        "neuro_updates": float(stats.get("neuro_updates", 0.0)),
                        "discovery_loss": float(stats.get("discovery_loss", 0.0)),
                        "discovery_alignment": float(stats.get("discovery_alignment", 0.0)),
                        "discovery_contrast": float(stats.get("discovery_contrast", 0.0)),
                        "discovery_imagination": float(stats.get("discovery_imagination", 0.0)),
                        "discovery_emotion": float(stats.get("discovery_emotion", 0.0)),
                        "discovery_confidence": float(stats.get("discovery_confidence", 0.0)),
                        "discovery_curiosity": float(stats.get("discovery_curiosity", 0.0)),
                        "discovery_counter_share": float(stats.get("discovery_counter_share", 0.0)),
                        "discovery_updates": float(stats.get("discovery_updates", 0.0)),
                        "transcendent_loss": float(stats.get("transcendent_loss", 0.0)),
                        "transcendent_stability": float(stats.get("transcendent_stability", 0.0)),
                        "transcendent_divergence": float(stats.get("transcendent_divergence", 0.0)),
                        "transcendent_foresight": float(stats.get("transcendent_foresight", 0.0)),
                        "transcendent_synthesis": float(stats.get("transcendent_synthesis", 0.0)),
                        "transcendent_affective": float(stats.get("transcendent_affective", 0.0)),
                        "transcendent_entropy": float(stats.get("transcendent_entropy", 0.0)),
                        "transcendent_coherence": float(stats.get("transcendent_coherence", 0.0)),
                        "transcendent_updates": float(stats.get("transcendent_updates", 0.0)),
                        "frontier_loss": float(stats.get("frontier_loss", 0.0)),
                        "frontier_novelty": float(stats.get("frontier_novelty", 0.0)),
                        "frontier_abstraction": float(stats.get("frontier_abstraction", 0.0)),
                        "frontier_transfer": float(stats.get("frontier_transfer", 0.0)),
                        "frontier_curiosity": float(stats.get("frontier_curiosity", 0.0)),
                        "frontier_emotion": float(stats.get("frontier_emotion", 0.0)),
                        "frontier_meta": float(stats.get("frontier_meta", 0.0)),
                        "frontier_diversity": float(stats.get("frontier_diversity", 0.0)),
                        "frontier_updates": float(stats.get("frontier_updates", 0.0)),
                    }
                )

                elapsed = time.perf_counter() - epoch_start
                print(
                    f"Fold {fold_index}/{total_folds} epoch {global_epoch:03d} [{stage_name}] "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"train_acc={train_acc * 100:.2f}% val_acc={val_acc * 100:.2f}% "
                    f"lr={current_lr:.6f} ({elapsed:.1f}s)"
                )
                if curriculum_summary:
                    hardest_examples = curriculum_summary.get("hardest_examples", [])
                    preview = "; ".join(
                        f"{item['label']}@{item['confidence']:.2f}→x{item['multiplier']:.2f}::{item['text']}"
                        for item in hardest_examples
                        if isinstance(item, dict)
                    )
                    if not preview:
                        preview = "n/a"
                    print(
                        f"   ↳ curriculum avg×{curriculum_summary['avg_multiplier']:.2f} "
                        f"(boosted {curriculum_summary['boosted']}, dampened {curriculum_summary['dampened']}, "
                        f"examples {curriculum_summary['examples']}); hardest {preview}"
                    )
                if fold_meta_config is not None and fold_meta_config.enabled:
                    print(
                        "   ↳ meta-introspection loss "
                        f"{stats.get('meta_loss', 0.0):.4f} "
                        f"gap {stats.get('meta_gap', 0.0):.3f} "
                        f"coverage {stats.get('meta_coverage', 0.0):.2f}"
                    )
                if fold_neuro_config is not None and fold_neuro_config.enabled:
                    print(
                        "   ↳ neuro-symbolic loss "
                        f"{stats.get('neuro_loss', 0.0):.4f} "
                        f"struct {stats.get('neuro_structural', 0.0):.4f} "
                        f"cohesion {stats.get('neuro_cohesion', 0.0):.3f} "
                        f"entropy {stats.get('neuro_entropy', 0.0):.3f}"
                    )
                if fold_discovery_config is not None and fold_discovery_config.enabled:
                    print(
                        "   ↳ self-discovery loss "
                        f"{stats.get('discovery_loss', 0.0):.4f} "
                        f"align {stats.get('discovery_alignment', 0.0):.4f} "
                        f"curiosity {stats.get('discovery_curiosity', 0.0):.3f}"
                    )
                if fold_transcendent_config is not None and fold_transcendent_config.enabled:
                    print(
                        "   ↳ transcendent cognition loss "
                        f"{stats.get('transcendent_loss', 0.0):.4f} "
                        f"coherence {stats.get('transcendent_coherence', 0.0):.3f} "
                        f"stability {stats.get('transcendent_stability', 0.0):.4f}"
                    )
                if fold_frontier_config is not None and fold_frontier_config.enabled:
                    print(
                        "   ↳ frontier intelligence loss "
                        f"{stats.get('frontier_loss', 0.0):.4f} "
                        f"novelty {stats.get('frontier_novelty', 0.0):.4f} "
                        f"diversity {stats.get('frontier_diversity', 0.0):.3f}"
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

        def maybe_run_self_play(stage_marker: str) -> bool:
            nonlocal epochs_since_improvement, total_self_play_added, self_play_round_counter
            if args.self_play_rounds <= 0 or args.self_play_per_label <= 0:
                return False
            pseudo_source: nn.Module = model
            if (
                ema_model is not None
                and args.ema_use_for_eval
                and ema_update_counter > 0
            ):
                pseudo_source = ema_model
            elif swa_model is not None and swa_update_counter > 0:
                pseudo_source = swa_model

            stage_stop = False
            for _ in range(args.self_play_rounds):
                self_play_round_counter += 1
                round_id = self_play_round_counter
                generators = build_label_ngram_models(
                    train_texts,
                    train_labels,
                    order=args.self_play_ngram_order,
                )
                if not generators:
                    if round_id == 1:
                        print(
                            f"Fold {fold_index}/{total_folds} self-play round {round_id}: "
                            "insufficient tokens to build label generators."
                        )
                    break

                attempted = 0
                accepted = 0
                rejected = 0
                mismatched = 0
                label_histogram: Counter[str] = Counter()
                accepted_confidences: List[float] = []
                accepted_consistency: List[float] = []
                accepted_margins: List[float] = []
                example_summaries: List[Dict[str, object]] = []
                aggregated_distribution: Dict[str, float] = defaultdict(float)

                label_items = list(generators.items())
                self_play_rng.shuffle(label_items)

                for source_label, generator in label_items:
                    if args.self_play_per_label <= 0:
                        continue
                    accepted_for_label = 0
                    attempts_for_label = 0
                    max_attempts = max(args.self_play_per_label * 5, args.self_play_per_label + 2)
                    while (
                        accepted_for_label < args.self_play_per_label
                        and attempts_for_label < max_attempts
                    ):
                        attempts_for_label += 1
                        tokens = sample_synthetic_tokens(
                            generator,
                            self_play_rng,
                            max_tokens=args.self_play_max_length,
                            temperature=args.self_play_temperature,
                        )
                        text = render_synthetic_text(tokens)
                        if not text or text in existing_texts:
                            continue
                        attempted += 1
                        evaluation = evaluate_self_play_candidate(
                            pseudo_source,
                            text,
                            vocab=vocab,
                            label_to_idx=label_to_idx,
                            max_len=max_seq_len,
                            device=device,
                            tokenizer=tokenizer_obj,
                            tokenizer_cache=tokenizer_cache_fn,
                            embedding_model=embedding_fn,
                            samples=args.self_play_samples,
                            emotion_encoder=emotion_lexicon if (emotion_enabled and emotion_dim > 0) else None,
                            emotion_dim=emotion_dim,
                            emotion_config=fold_emotion_config,
                        )
                        if evaluation is None:
                            rejected += 1
                            continue
                        predicted_label = evaluation.label
                        if args.self_play_require_match and predicted_label != source_label:
                            mismatched += 1
                            rejected += 1
                            continue
                        if (
                            evaluation.consistency < args.self_play_consistency
                            or evaluation.blended_confidence < args.self_play_min_confidence
                        ):
                            rejected += 1
                            continue
                        weight = compute_pseudo_weight(
                            float(args.self_play_weight),
                            float(evaluation.blended_confidence),
                            float(args.self_play_min_confidence),
                            float(args.self_play_confidence_power),
                            float(args.self_play_max_weight_multiplier),
                        )
                        train_texts.append(text)
                        train_labels.append(predicted_label)
                        train_weights.append(weight)
                        if emotion_enabled and emotion_lexicon is not None and fold_emotion_memory is not None and emotion_dim > 0:
                            vector = list(emotion_lexicon.vectorise(text))
                            train_emotion_vectors.append(vector)
                            fold_emotion_memory.register_vectors(
                                [label_to_idx[predicted_label]],
                                [vector],
                                weights=[weight],
                            )
                        elif emotion_enabled and fold_emotion_memory is not None:
                            train_emotion_vectors.append([0.0] * emotion_dim)
                        if curriculum_manager is not None:
                            curriculum_manager.register_samples([text], [weight])
                        existing_texts.add(text)
                        self_play_examples_store.append((text, predicted_label, weight))
                        total_self_play_added += 1
                        accepted += 1
                        accepted_for_label += 1
                        label_histogram[predicted_label] += 1
                        accepted_confidences.append(evaluation.blended_confidence)
                        accepted_consistency.append(evaluation.consistency)
                        accepted_margins.append(evaluation.margin)
                        for label, value in evaluation.average_distribution.items():
                            aggregated_distribution[label] += value
                        snippet = text if len(text) <= 160 else text[:157] + "..."
                        example_summaries.append(
                            {
                                "text": snippet,
                                "predicted_label": predicted_label,
                                "confidence": evaluation.blended_confidence,
                                "deterministic_confidence": evaluation.deterministic_confidence,
                                "mc_confidence": evaluation.mc_confidence,
                                "consistency": evaluation.consistency,
                                "margin": evaluation.margin,
                                "source_label": source_label,
                                "top_predictions": [
                                    {"label": label, "confidence": score}
                                    for label, score in evaluation.top_predictions
                                ],
                            }
                        )

                mean_conf = sum(accepted_confidences) / len(accepted_confidences) if accepted_confidences else 0.0
                mean_consistency = sum(accepted_consistency) / len(accepted_consistency) if accepted_consistency else 0.0
                mean_margin = sum(accepted_margins) / len(accepted_margins) if accepted_margins else 0.0
                distribution_summary = {
                    label: value / accepted if accepted else 0.0
                    for label, value in aggregated_distribution.items()
                }
                self_play_rounds.append(
                    {
                        "round": float(round_id),
                        "stage": stage_marker,
                        "attempted": float(attempted),
                        "accepted": float(accepted),
                        "rejected": float(rejected),
                        "mismatched": float(mismatched),
                        "mean_confidence": float(mean_conf),
                        "mean_consistency": float(mean_consistency),
                        "mean_margin": float(mean_margin),
                        "label_histogram": dict(sorted(label_histogram.items())),
                        "average_distribution": distribution_summary,
                        "examples": example_summaries[:5],
                    }
                )

                if accepted > 0:
                    print(
                        f"Fold {fold_index}/{total_folds} self-play round {round_id}: "
                        f"accepted {accepted} synthetic examples (avg confidence {mean_conf:.3f}, "
                        f"consistency {mean_consistency:.3f})."
                    )
                    if args.self_play_epochs > 0:
                        epochs_since_improvement = 0
                        stage_stop = run_stage(f"self-play-{round_id}", args.self_play_epochs)
                        if stage_stop:
                            break
                else:
                    print(
                        f"Fold {fold_index}/{total_folds} self-play round {round_id}: "
                        "no synthetic examples met the acceptance criteria."
                    )
            return stage_stop

        stop_training = run_stage("supervised", args.epochs)

        if not stop_training:
            stop_training = maybe_run_self_play("supervised")

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
                    emotion_encoder=emotion_lexicon if (emotion_enabled and emotion_dim > 0) else None,
                    emotion_dim=emotion_dim,
                    emotion_config=fold_emotion_config,
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
                    if emotion_enabled and emotion_lexicon is not None and fold_emotion_memory is not None and emotion_dim > 0:
                        vector = list(emotion_lexicon.vectorise(text))
                        train_emotion_vectors.append(vector)
                        fold_emotion_memory.register_vectors(
                            [label_to_idx[label]],
                            [vector],
                            weights=[weight],
                        )
                    elif emotion_enabled and fold_emotion_memory is not None:
                        train_emotion_vectors.append([0.0] * emotion_dim)
                    if curriculum_manager is not None:
                        curriculum_manager.register_samples([text], [weight])
                    existing_texts.add(text)
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
                if not stop_training:
                    stop_training = maybe_run_self_play(f"self-train-{round_idx}")

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

        meta_snapshot: Optional[Dict[str, object]] = None
        if fold_meta_config is not None and fold_meta_config.enabled:
            meta_snapshot = fold_meta_config.introspector.snapshot()

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
            "self_play": {
                "rounds_configured": args.self_play_rounds,
                "rounds_ran": len(self_play_rounds),
                "epochs_per_round": args.self_play_epochs,
                "per_label_target": args.self_play_per_label,
                "max_length": args.self_play_max_length,
                "samples": args.self_play_samples,
                "min_confidence": args.self_play_min_confidence,
                "consistency_threshold": args.self_play_consistency,
                "weight": args.self_play_weight,
                "confidence_power": args.self_play_confidence_power,
                "max_weight_multiplier": args.self_play_max_weight_multiplier,
                "temperature": args.self_play_temperature,
                "ngram_order": args.self_play_ngram_order,
                "require_label_match": bool(args.self_play_require_match),
                "examples_added": total_self_play_added,
                "round_details": self_play_rounds,
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
            "adaptive_curriculum": (
                curriculum_manager.export_metadata()
                if curriculum_manager is not None
                else {"enabled": False}
            ),
            "emotion_reasoner": (
                {
                    "enabled": True,
                    "dimension": emotion_dim,
                    "consistency_weight": args.emotion_consistency_weight,
                    "temperature": args.emotion_expectation_temperature,
                    "prototype_smoothing": args.emotion_prototype_smoothing,
                    "fusion_dropout": args.emotion_fusion_dropout,
                    "memory_updates": int(fold_emotion_memory.total_updates),
                }
                if emotion_enabled and fold_emotion_memory is not None and emotion_dim > 0
                else {"enabled": False}
            ),
            "meta_introspection": (
                {
                    "enabled": True,
                    "attraction_weight": fold_meta_config.attraction_weight,
                    "repulsion_weight": fold_meta_config.repulsion_weight,
                    "discovery_weight": fold_meta_config.discovery_weight,
                    "gap_margin": fold_meta_config.gap_margin,
                    "temperature": fold_meta_config.temperature,
                    "momentum": fold_meta_config.introspector.momentum,
                    "margin": fold_meta_config.introspector.margin,
                    "history_limit": fold_meta_config.introspector.history.maxlen,
                    "snapshot": meta_snapshot,
                }
                if fold_meta_config is not None and fold_meta_config.enabled
                else {"enabled": False}
            ),
            "neuro_symbolic": (
                {
                    "enabled": True,
                    "structural_weight": fold_neuro_config.structural_weight,
                    "semantic_weight": fold_neuro_config.semantic_weight,
                    "affective_weight": fold_neuro_config.affective_weight,
                    "temperature": fold_neuro_config.temperature,
                    "self_loop": fold_neuro_config.self_loop,
                    "lexical_weight": fold_neuro_config.reasoner.lexical_weight,
                    "graph_momentum": fold_neuro_config.reasoner.graph_momentum,
                    "feature_momentum": fold_neuro_config.reasoner.feature_momentum,
                    "min_confidence": fold_neuro_config.reasoner.min_confidence,
                    "history_limit": fold_neuro_config.reasoner.history.maxlen,
                    "snapshot": fold_neuro_config.reasoner.snapshot(),
                    "metadata": fold_neuro_config.reasoner.export_metadata(),
                }
                if fold_neuro_config is not None and fold_neuro_config.enabled
                else {"enabled": False}
            ),
            "self_discovery": (
                {
                    "enabled": True,
                    "alignment_weight": fold_discovery_config.alignment_weight,
                    "contrast_weight": fold_discovery_config.contrast_weight,
                    "imagination_weight": fold_discovery_config.imagination_weight,
                    "emotion_weight": fold_discovery_config.emotion_weight,
                    "temperature": fold_discovery_config.temperature,
                    "min_confidence": fold_discovery_config.min_confidence,
                    "margin": fold_discovery_config.margin,
                    "feature_momentum": fold_discovery_config.orchestrator.feature_momentum,
                    "counter_momentum": fold_discovery_config.orchestrator.counter_momentum,
                    "imagination_momentum": fold_discovery_config.orchestrator.imagination_momentum,
                    "curiosity_weight": fold_discovery_config.orchestrator.curiosity_weight,
                    "history_limit": fold_discovery_config.orchestrator.history.maxlen,
                    "snapshot": fold_discovery_config.orchestrator.snapshot(),
                    "metadata": fold_discovery_config.orchestrator.export_metadata(),
                }
                if fold_discovery_config is not None and fold_discovery_config.enabled
                else {"enabled": bool(args.self_discovery)}
            ),
            "transcendent_cognition": (
                {
                    "enabled": True,
                    "stability_weight": fold_transcendent_config.stability_weight,
                    "divergence_weight": fold_transcendent_config.divergence_weight,
                    "foresight_weight": fold_transcendent_config.foresight_weight,
                    "synthesis_weight": fold_transcendent_config.synthesis_weight,
                    "affective_weight": fold_transcendent_config.affective_weight,
                    "entropy_weight": fold_transcendent_config.entropy_weight,
                    "temperature": fold_transcendent_config.temperature,
                    "margin": fold_transcendent_config.margin,
                    "feature_momentum": fold_transcendent_config.architect.feature_momentum,
                    "counter_momentum": fold_transcendent_config.architect.counter_momentum,
                    "transition_momentum": fold_transcendent_config.architect.transition_momentum,
                    "imagination_momentum": fold_transcendent_config.architect.imagination_momentum,
                    "max_glimpses": fold_transcendent_config.architect.max_glimpses,
                    "history_limit": fold_transcendent_config.architect.history.maxlen,
                    "snapshot": fold_transcendent_config.architect.snapshot(),
                    "metadata": fold_transcendent_config.architect.export_metadata(),
                }
                if fold_transcendent_config is not None and fold_transcendent_config.enabled
                else {"enabled": bool(args.transcendent_cognition)}
            ),
            "frontier_intelligence": (
                {
                    "enabled": True,
                    "novelty_weight": fold_frontier_config.novelty_weight,
                    "abstraction_weight": fold_frontier_config.abstraction_weight,
                    "transfer_weight": fold_frontier_config.transfer_weight,
                    "curiosity_weight": fold_frontier_config.curiosity_weight,
                    "emotion_weight": fold_frontier_config.emotion_weight,
                    "meta_weight": fold_frontier_config.meta_weight,
                    "temperature": fold_frontier_config.temperature,
                    "margin": fold_frontier_config.margin,
                    "concept_momentum": fold_frontier_config.catalyst.concept_momentum,
                    "bridge_momentum": fold_frontier_config.catalyst.bridge_momentum,
                    "novelty_momentum": fold_frontier_config.catalyst.novelty_momentum,
                    "meta_momentum": fold_frontier_config.catalyst.meta_momentum,
                    "emotion_momentum": fold_frontier_config.catalyst.emotion_momentum,
                    "history_limit": fold_frontier_config.catalyst.history.maxlen,
                    "snapshot": fold_frontier_config.catalyst.snapshot(),
                    "metadata": fold_frontier_config.catalyst.export_metadata(),
                }
                if fold_frontier_config is not None and fold_frontier_config.enabled
                else {"enabled": bool(args.frontier_intelligence)}
            ),
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

        emotion_alignment_values = [
            float(entry.get("emotion_alignment", 0.0))
            for entry in history
            if "emotion_alignment" in entry
        ]
        meta_loss_values = [
            float(entry.get("meta_loss", 0.0))
            for entry in history
            if "meta_loss" in entry
        ]
        meta_gap_values = [
            float(entry.get("meta_gap", 0.0))
            for entry in history
            if "meta_gap" in entry
        ]
        meta_entropy_values = [
            float(entry.get("meta_entropy", 0.0))
            for entry in history
            if "meta_entropy" in entry
        ]
        meta_coverage_values = [
            float(entry.get("meta_coverage", 0.0))
            for entry in history
            if "meta_coverage" in entry
        ]
        meta_update_values = [
            float(entry.get("meta_updates", 0.0))
            for entry in history
            if "meta_updates" in entry
        ]
        discovery_loss_values = [
            float(entry.get("discovery_loss", 0.0))
            for entry in history
            if "discovery_loss" in entry
        ]
        discovery_alignment_values = [
            float(entry.get("discovery_alignment", 0.0))
            for entry in history
            if "discovery_alignment" in entry
        ]
        discovery_contrast_values = [
            float(entry.get("discovery_contrast", 0.0))
            for entry in history
            if "discovery_contrast" in entry
        ]
        discovery_imagination_values = [
            float(entry.get("discovery_imagination", 0.0))
            for entry in history
            if "discovery_imagination" in entry
        ]
        discovery_emotion_values = [
            float(entry.get("discovery_emotion", 0.0))
            for entry in history
            if "discovery_emotion" in entry
        ]
        discovery_confidence_values = [
            float(entry.get("discovery_confidence", 0.0))
            for entry in history
            if "discovery_confidence" in entry
        ]
        discovery_curiosity_values = [
            float(entry.get("discovery_curiosity", 0.0))
            for entry in history
            if "discovery_curiosity" in entry
        ]
        discovery_counter_values = [
            float(entry.get("discovery_counter_share", 0.0))
            for entry in history
            if "discovery_counter_share" in entry
        ]
        discovery_update_values = [
            float(entry.get("discovery_updates", 0.0))
            for entry in history
            if "discovery_updates" in entry
        ]
        transcendent_loss_values = [
            float(entry.get("transcendent_loss", 0.0))
            for entry in history
            if "transcendent_loss" in entry
        ]
        transcendent_stability_values = [
            float(entry.get("transcendent_stability", 0.0))
            for entry in history
            if "transcendent_stability" in entry
        ]
        transcendent_divergence_values = [
            float(entry.get("transcendent_divergence", 0.0))
            for entry in history
            if "transcendent_divergence" in entry
        ]
        transcendent_foresight_values = [
            float(entry.get("transcendent_foresight", 0.0))
            for entry in history
            if "transcendent_foresight" in entry
        ]
        transcendent_synthesis_values = [
            float(entry.get("transcendent_synthesis", 0.0))
            for entry in history
            if "transcendent_synthesis" in entry
        ]
        transcendent_affective_values = [
            float(entry.get("transcendent_affective", 0.0))
            for entry in history
            if "transcendent_affective" in entry
        ]
        transcendent_entropy_values = [
            float(entry.get("transcendent_entropy", 0.0))
            for entry in history
            if "transcendent_entropy" in entry
        ]
        transcendent_coherence_values = [
            float(entry.get("transcendent_coherence", 0.0))
            for entry in history
            if "transcendent_coherence" in entry
        ]
        transcendent_update_values = [
            float(entry.get("transcendent_updates", 0.0))
            for entry in history
            if "transcendent_updates" in entry
        ]
        frontier_loss_values = [
            float(entry.get("frontier_loss", 0.0))
            for entry in history
            if "frontier_loss" in entry
        ]
        frontier_novelty_values = [
            float(entry.get("frontier_novelty", 0.0))
            for entry in history
            if "frontier_novelty" in entry
        ]
        frontier_abstraction_values = [
            float(entry.get("frontier_abstraction", 0.0))
            for entry in history
            if "frontier_abstraction" in entry
        ]
        frontier_transfer_values = [
            float(entry.get("frontier_transfer", 0.0))
            for entry in history
            if "frontier_transfer" in entry
        ]
        frontier_curiosity_values = [
            float(entry.get("frontier_curiosity", 0.0))
            for entry in history
            if "frontier_curiosity" in entry
        ]
        frontier_emotion_values = [
            float(entry.get("frontier_emotion", 0.0))
            for entry in history
            if "frontier_emotion" in entry
        ]
        frontier_meta_values = [
            float(entry.get("frontier_meta", 0.0))
            for entry in history
            if "frontier_meta" in entry
        ]
        frontier_diversity_values = [
            float(entry.get("frontier_diversity", 0.0))
            for entry in history
            if "frontier_diversity" in entry
        ]
        frontier_update_values = [
            float(entry.get("frontier_updates", 0.0))
            for entry in history
            if "frontier_updates" in entry
        ]
        neuro_loss_values = [
            float(entry.get("neuro_loss", 0.0))
            for entry in history
            if "neuro_loss" in entry
        ]
        neuro_struct_values = [
            float(entry.get("neuro_structural", 0.0))
            for entry in history
            if "neuro_structural" in entry
        ]
        neuro_semantic_values = [
            float(entry.get("neuro_semantic", 0.0))
            for entry in history
            if "neuro_semantic" in entry
        ]
        neuro_affective_values = [
            float(entry.get("neuro_affective", 0.0))
            for entry in history
            if "neuro_affective" in entry
        ]
        neuro_entropy_values = [
            float(entry.get("neuro_entropy", 0.0))
            for entry in history
            if "neuro_entropy" in entry
        ]
        neuro_cohesion_values = [
            float(entry.get("neuro_cohesion", 0.0))
            for entry in history
            if "neuro_cohesion" in entry
        ]
        neuro_update_values = [
            float(entry.get("neuro_updates", 0.0))
            for entry in history
            if "neuro_updates" in entry
        ]
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
            "synthetic_examples_added": total_self_play_added,
            "remaining_unlabeled": len(unlabeled_texts),
            "self_training_rounds_configured": args.self_train_rounds,
            "self_training_rounds_completed": len(pseudo_rounds),
            "self_play_rounds_configured": args.self_play_rounds,
            "self_play_rounds_completed": len(self_play_rounds),
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
        metrics["emotion_reasoner_enabled"] = bool(emotion_enabled and emotion_dim > 0)
        if emotion_enabled and fold_emotion_memory is not None and emotion_dim > 0:
            metrics["emotion_memory_updates"] = int(fold_emotion_memory.total_updates)
        if emotion_enabled and emotion_alignment_values:
            metrics["emotion_alignment_mean"] = float(
                sum(emotion_alignment_values) / len(emotion_alignment_values)
            )
            metrics["emotion_alignment_last"] = float(emotion_alignment_values[-1])
        metrics["meta_introspection_enabled"] = bool(fold_meta_config is not None and fold_meta_config.enabled)
        if meta_loss_values:
            metrics["meta_loss_mean"] = float(sum(meta_loss_values) / len(meta_loss_values))
            metrics["meta_loss_last"] = float(meta_loss_values[-1])
        if meta_gap_values:
            metrics["meta_gap_mean"] = float(sum(meta_gap_values) / len(meta_gap_values))
            metrics["meta_gap_last"] = float(meta_gap_values[-1])
        if meta_entropy_values:
            metrics["meta_entropy_mean"] = float(
                sum(meta_entropy_values) / len(meta_entropy_values)
            )
            metrics["meta_entropy_last"] = float(meta_entropy_values[-1])
        if meta_coverage_values:
            metrics["meta_coverage_mean"] = float(
                sum(meta_coverage_values) / len(meta_coverage_values)
            )
            metrics["meta_coverage_last"] = float(meta_coverage_values[-1])
        if meta_update_values:
            metrics["meta_updates_last"] = float(meta_update_values[-1])
        metrics["self_discovery_enabled"] = bool(
            fold_discovery_config is not None and fold_discovery_config.enabled
        )
        if discovery_loss_values:
            metrics["discovery_loss_mean"] = float(
                sum(discovery_loss_values) / len(discovery_loss_values)
            )
            metrics["discovery_loss_last"] = float(discovery_loss_values[-1])
        if discovery_alignment_values:
            metrics["discovery_alignment_mean"] = float(
                sum(discovery_alignment_values) / len(discovery_alignment_values)
            )
            metrics["discovery_alignment_last"] = float(discovery_alignment_values[-1])
        if discovery_contrast_values:
            metrics["discovery_contrast_mean"] = float(
                sum(discovery_contrast_values) / len(discovery_contrast_values)
            )
            metrics["discovery_contrast_last"] = float(discovery_contrast_values[-1])
        if discovery_imagination_values:
            metrics["discovery_imagination_mean"] = float(
                sum(discovery_imagination_values) / len(discovery_imagination_values)
            )
            metrics["discovery_imagination_last"] = float(discovery_imagination_values[-1])
        if discovery_emotion_values:
            metrics["discovery_emotion_mean"] = float(
                sum(discovery_emotion_values) / len(discovery_emotion_values)
            )
            metrics["discovery_emotion_last"] = float(discovery_emotion_values[-1])
        if discovery_confidence_values:
            metrics["discovery_confidence_mean"] = float(
                sum(discovery_confidence_values) / len(discovery_confidence_values)
            )
            metrics["discovery_confidence_last"] = float(discovery_confidence_values[-1])
        if discovery_curiosity_values:
            metrics["discovery_curiosity_mean"] = float(
                sum(discovery_curiosity_values) / len(discovery_curiosity_values)
            )
            metrics["discovery_curiosity_last"] = float(discovery_curiosity_values[-1])
        if discovery_counter_values:
            metrics["discovery_counter_share_mean"] = float(
                sum(discovery_counter_values) / len(discovery_counter_values)
            )
            metrics["discovery_counter_share_last"] = float(discovery_counter_values[-1])
        if discovery_update_values:
            metrics["discovery_updates_last"] = float(discovery_update_values[-1])
        metrics["transcendent_cognition_enabled"] = bool(
            fold_transcendent_config is not None and fold_transcendent_config.enabled
        )
        if transcendent_loss_values:
            metrics["transcendent_loss_mean"] = float(
                sum(transcendent_loss_values) / len(transcendent_loss_values)
            )
            metrics["transcendent_loss_last"] = float(transcendent_loss_values[-1])
        if transcendent_stability_values:
            metrics["transcendent_stability_mean"] = float(
                sum(transcendent_stability_values) / len(transcendent_stability_values)
            )
            metrics["transcendent_stability_last"] = float(transcendent_stability_values[-1])
        if transcendent_divergence_values:
            metrics["transcendent_divergence_mean"] = float(
                sum(transcendent_divergence_values) / len(transcendent_divergence_values)
            )
            metrics["transcendent_divergence_last"] = float(transcendent_divergence_values[-1])
        if transcendent_foresight_values:
            metrics["transcendent_foresight_mean"] = float(
                sum(transcendent_foresight_values) / len(transcendent_foresight_values)
            )
            metrics["transcendent_foresight_last"] = float(transcendent_foresight_values[-1])
        if transcendent_synthesis_values:
            metrics["transcendent_synthesis_mean"] = float(
                sum(transcendent_synthesis_values) / len(transcendent_synthesis_values)
            )
            metrics["transcendent_synthesis_last"] = float(transcendent_synthesis_values[-1])
        if transcendent_affective_values:
            metrics["transcendent_affective_mean"] = float(
                sum(transcendent_affective_values) / len(transcendent_affective_values)
            )
            metrics["transcendent_affective_last"] = float(transcendent_affective_values[-1])
        if transcendent_entropy_values:
            metrics["transcendent_entropy_mean"] = float(
                sum(transcendent_entropy_values) / len(transcendent_entropy_values)
            )
            metrics["transcendent_entropy_last"] = float(transcendent_entropy_values[-1])
        if transcendent_coherence_values:
            metrics["transcendent_coherence_mean"] = float(
                sum(transcendent_coherence_values) / len(transcendent_coherence_values)
            )
            metrics["transcendent_coherence_last"] = float(transcendent_coherence_values[-1])
        if transcendent_update_values:
            metrics["transcendent_updates_last"] = float(transcendent_update_values[-1])
        metrics["frontier_intelligence_enabled"] = bool(
            fold_frontier_config is not None and fold_frontier_config.enabled
        )
        if frontier_loss_values:
            metrics["frontier_loss_mean"] = float(
                sum(frontier_loss_values) / len(frontier_loss_values)
            )
            metrics["frontier_loss_last"] = float(frontier_loss_values[-1])
        if frontier_novelty_values:
            metrics["frontier_novelty_mean"] = float(
                sum(frontier_novelty_values) / len(frontier_novelty_values)
            )
            metrics["frontier_novelty_last"] = float(frontier_novelty_values[-1])
        if frontier_abstraction_values:
            metrics["frontier_abstraction_mean"] = float(
                sum(frontier_abstraction_values) / len(frontier_abstraction_values)
            )
            metrics["frontier_abstraction_last"] = float(frontier_abstraction_values[-1])
        if frontier_transfer_values:
            metrics["frontier_transfer_mean"] = float(
                sum(frontier_transfer_values) / len(frontier_transfer_values)
            )
            metrics["frontier_transfer_last"] = float(frontier_transfer_values[-1])
        if frontier_curiosity_values:
            metrics["frontier_curiosity_mean"] = float(
                sum(frontier_curiosity_values) / len(frontier_curiosity_values)
            )
            metrics["frontier_curiosity_last"] = float(frontier_curiosity_values[-1])
        if frontier_emotion_values:
            metrics["frontier_emotion_mean"] = float(
                sum(frontier_emotion_values) / len(frontier_emotion_values)
            )
            metrics["frontier_emotion_last"] = float(frontier_emotion_values[-1])
        if frontier_meta_values:
            metrics["frontier_meta_mean"] = float(
                sum(frontier_meta_values) / len(frontier_meta_values)
            )
            metrics["frontier_meta_last"] = float(frontier_meta_values[-1])
        if frontier_diversity_values:
            metrics["frontier_diversity_mean"] = float(
                sum(frontier_diversity_values) / len(frontier_diversity_values)
            )
            metrics["frontier_diversity_last"] = float(frontier_diversity_values[-1])
        if frontier_update_values:
            metrics["frontier_updates_last"] = float(frontier_update_values[-1])
        metrics["neuro_symbolic_enabled"] = bool(fold_neuro_config is not None and fold_neuro_config.enabled)
        if neuro_loss_values:
            metrics["neuro_loss_mean"] = float(sum(neuro_loss_values) / len(neuro_loss_values))
            metrics["neuro_loss_last"] = float(neuro_loss_values[-1])
        if neuro_struct_values:
            metrics["neuro_structural_mean"] = float(sum(neuro_struct_values) / len(neuro_struct_values))
            metrics["neuro_structural_last"] = float(neuro_struct_values[-1])
        if neuro_semantic_values:
            metrics["neuro_semantic_mean"] = float(sum(neuro_semantic_values) / len(neuro_semantic_values))
            metrics["neuro_semantic_last"] = float(neuro_semantic_values[-1])
        if neuro_affective_values:
            metrics["neuro_affective_mean"] = float(sum(neuro_affective_values) / len(neuro_affective_values))
            metrics["neuro_affective_last"] = float(neuro_affective_values[-1])
        if neuro_entropy_values:
            metrics["neuro_entropy_mean"] = float(sum(neuro_entropy_values) / len(neuro_entropy_values))
            metrics["neuro_entropy_last"] = float(neuro_entropy_values[-1])
        if neuro_cohesion_values:
            metrics["neuro_cohesion_mean"] = float(sum(neuro_cohesion_values) / len(neuro_cohesion_values))
            metrics["neuro_cohesion_last"] = float(neuro_cohesion_values[-1])
        if neuro_update_values:
            metrics["neuro_updates_last"] = float(neuro_update_values[-1])

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
        if curriculum_manager is not None:
            metrics.update(curriculum_manager.export_metrics())

        evaluation_outputs: List[Dict[str, object]] = []
        for sample in evaluation_inputs:
            prediction = predict_with_trace(
                model,
                sample,
                vocab=vocab,
                label_to_idx=label_to_idx,
                max_len=max_seq_len,
                device=device,
                tokenizer=tokenizer_obj,
                tokenizer_cache=tokenizer_cache_fn,
                embedding_model=embedding_fn,
                emotion_encoder=emotion_lexicon if (emotion_enabled and emotion_dim > 0) else None,
                emotion_dim=emotion_dim,
                emotion_config=fold_emotion_config,
            )
            response = generate_response(prediction.label, sample)
            entry: Dict[str, object] = {
                "input": sample,
                "predicted_intent": prediction.label,
                "confidence": prediction.confidence,
                "top_intents": [
                    {"label": label, "confidence": score}
                    for label, score in prediction.top_predictions
                ],
                "response": response.message,
                "response_strategy": response.strategy,
            }
            if response.basis:
                entry["response_basis"] = response.basis
            evaluation_outputs.append(entry)

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
            self_play_rounds=self_play_rounds,
            total_self_play_added=total_self_play_added,
            self_play_examples=self_play_examples_store,
            model_state=best_state,
            evaluation_outputs=evaluation_outputs,
            run_tag_suffix=fold_suffix,
        )
    def run_full_dataset_training(best_fold: FoldResult) -> Optional[Dict[str, object]]:
        nonlocal fp16_warning_emitted
        if args.final_train_epochs <= 0:
            return None

        final_texts = list(texts)
        final_labels = list(labels)
        final_weights = [1.0] * len(final_texts)
        final_emotion_vectors: List[List[float]] = []
        final_emotion_memory: Optional[EmotionPrototypeMemory] = None
        final_emotion_config: Optional[EmotionTrainingConfig] = None
        if emotion_enabled and emotion_lexicon is not None and emotion_dim > 0:
            final_emotion_vectors = [list(emotion_lexicon.vectorise(text)) for text in final_texts]
            final_emotion_memory = EmotionPrototypeMemory(
                num_classes,
                emotion_dim,
                smoothing=args.emotion_prototype_smoothing,
            )
            final_emotion_memory.register_vectors(
                [label_to_idx[label] for label in final_labels],
                final_emotion_vectors,
                weights=final_weights,
            )
            final_emotion_config = EmotionTrainingConfig(
                memory=final_emotion_memory,
                weight=args.emotion_consistency_weight,
                temperature=args.emotion_expectation_temperature,
                enabled=emotion_enabled,
            )
        final_meta_config: Optional[MetaCognitiveConfig] = None
        if args.meta_introspector:
            final_meta_config = MetaCognitiveConfig(
                introspector=MetaCognitiveIntrospector(
                    num_classes,
                    momentum=args.meta_momentum,
                    margin=args.meta_margin,
                    history_limit=args.meta_history,
                ),
                enabled=True,
                attraction_weight=args.meta_attraction_weight,
                repulsion_weight=args.meta_repulsion_weight,
                discovery_weight=args.meta_discovery_weight,
                gap_margin=args.meta_min_confidence_gap,
                temperature=args.meta_temperature,
            )
            best_meta_snapshot = None
            meta_section = best_fold.metadata.get("meta_introspection") if isinstance(best_fold.metadata, dict) else None
            if isinstance(meta_section, dict):
                best_meta_snapshot = meta_section.get("snapshot")
            if isinstance(best_meta_snapshot, dict):
                try:
                    final_meta_config.introspector.load_snapshot(best_meta_snapshot, device)
                except Exception:
                    pass
        final_neuro_config: Optional[NeuroSymbolicConfig] = None
        if args.neuro_symbolic_reasoner:
            lexical_profiles, lexical_keywords = build_label_concept_library(
                final_texts,
                final_labels,
                label_to_idx=label_to_idx,
                max_keywords=args.neuro_max_keywords,
            )
            final_neuro_config = NeuroSymbolicConfig(
                reasoner=NeuroSymbolicReasoner(
                    num_classes,
                    idx_to_label=idx_to_label_list,
                    lexical_profiles=lexical_profiles,
                    lexical_keywords=lexical_keywords,
                    lexical_weight=args.neuro_lexical_weight,
                    graph_momentum=args.neuro_graph_momentum,
                    feature_momentum=args.neuro_feature_momentum,
                    min_confidence=args.neuro_min_confidence,
                    history_limit=args.neuro_history,
                    emotion_dim=emotion_dim if emotion_enabled else 0,
                ),
                enabled=True,
                structural_weight=args.neuro_structural_weight,
                semantic_weight=args.neuro_semantic_weight,
                affective_weight=args.neuro_affective_weight,
                temperature=args.neuro_temperature,
                self_loop=args.neuro_self_loop,
            )
            neuro_section = best_fold.metadata.get("neuro_symbolic") if isinstance(best_fold.metadata, dict) else None
            best_neuro_snapshot = None
            if isinstance(neuro_section, dict):
                best_neuro_snapshot = neuro_section.get("snapshot")
            if isinstance(best_neuro_snapshot, dict):
                try:
                    final_neuro_config.reasoner.load_snapshot(best_neuro_snapshot)
                except Exception:
                    pass
        final_discovery_config: Optional[SelfDiscoveryConfig] = None
        if args.self_discovery:
            final_discovery_config = SelfDiscoveryConfig(
                orchestrator=SelfDiscoveryOrchestrator(
                    num_classes,
                    feature_momentum=args.discovery_feature_momentum,
                    counter_momentum=args.discovery_counter_momentum,
                    imagination_momentum=args.discovery_imagination_momentum,
                    curiosity_weight=args.discovery_curiosity_weight,
                    history_limit=args.discovery_history,
                ),
                enabled=True,
                alignment_weight=args.discovery_alignment_weight,
                contrast_weight=args.discovery_contrast_weight,
                imagination_weight=args.discovery_imagination_weight,
                emotion_weight=args.discovery_emotion_weight,
                temperature=args.discovery_temperature,
                min_confidence=args.discovery_min_confidence,
                margin=args.discovery_margin,
            )
            discovery_section = best_fold.metadata.get("self_discovery") if isinstance(best_fold.metadata, dict) else None
            best_discovery_snapshot = None
            if isinstance(discovery_section, dict):
                best_discovery_snapshot = discovery_section.get("snapshot")
            if isinstance(best_discovery_snapshot, dict):
                try:
                    final_discovery_config.orchestrator.load_snapshot(best_discovery_snapshot)
                except Exception:
                    pass
        final_transcendent_config: Optional[TranscendentCognitionConfig] = None
        if args.transcendent_cognition:
            final_transcendent_config = TranscendentCognitionConfig(
                architect=TranscendentCognitionEngine(
                    num_classes,
                    feature_momentum=args.transcendent_feature_momentum,
                    counter_momentum=args.transcendent_counter_momentum,
                    transition_momentum=args.transcendent_transition_momentum,
                    imagination_momentum=args.transcendent_imagination_momentum,
                    history_limit=args.transcendent_history,
                    max_glimpses=args.transcendent_max_glimpses,
                ),
                enabled=True,
                stability_weight=args.transcendent_stability_weight,
                divergence_weight=args.transcendent_divergence_weight,
                foresight_weight=args.transcendent_foresight_weight,
                synthesis_weight=args.transcendent_synthesis_weight,
                affective_weight=args.transcendent_affective_weight,
                entropy_weight=args.transcendent_entropy_weight,
                temperature=args.transcendent_temperature,
                margin=args.transcendent_margin,
            )
            transcendent_section = (
                best_fold.metadata.get("transcendent_cognition")
                if isinstance(best_fold.metadata, dict)
                else None
            )
            best_transcendent_snapshot = None
            if isinstance(transcendent_section, dict):
                best_transcendent_snapshot = transcendent_section.get("snapshot")
            if isinstance(best_transcendent_snapshot, dict):
                try:
                    final_transcendent_config.architect.load_snapshot(best_transcendent_snapshot)
                except Exception:
                    pass
        final_frontier_config: Optional[FrontierIntelligenceConfig] = None
        if args.frontier_intelligence:
            final_frontier_config = FrontierIntelligenceConfig(
                catalyst=FrontierIntelligenceEngine(
                    num_classes,
                    concept_momentum=args.frontier_concept_momentum,
                    bridge_momentum=args.frontier_bridge_momentum,
                    novelty_momentum=args.frontier_novelty_momentum,
                    meta_momentum=args.frontier_meta_momentum,
                    emotion_momentum=args.frontier_emotion_momentum,
                    history_limit=args.frontier_history,
                ),
                enabled=True,
                novelty_weight=args.frontier_novelty_weight,
                abstraction_weight=args.frontier_abstraction_weight,
                transfer_weight=args.frontier_transfer_weight,
                curiosity_weight=args.frontier_curiosity_weight,
                emotion_weight=args.frontier_emotion_weight,
                meta_weight=args.frontier_meta_weight,
                temperature=args.frontier_temperature,
                margin=args.frontier_margin,
            )
            frontier_section = (
                best_fold.metadata.get("frontier_intelligence")
                if isinstance(best_fold.metadata, dict)
                else None
            )
            best_frontier_snapshot = None
            if isinstance(frontier_section, dict):
                best_frontier_snapshot = frontier_section.get("snapshot")
            if isinstance(best_frontier_snapshot, dict):
                try:
                    final_frontier_config.catalyst.load_snapshot(best_frontier_snapshot)
                except Exception:
                    pass
        pseudo_total = 0
        synthetic_total = 0
        if args.final_use_pseudo:
            pseudo_cache: Dict[str, Tuple[str, float]] = {}
            pseudo_seen: Set[str] = set()
            synthetic_seen: Set[str] = set()
            for result in fold_results:
                for text, label, weight in result.pseudo_examples:
                    pseudo_seen.add(text)
                    existing = pseudo_cache.get(text)
                    if existing is None or existing[1] < weight:
                        pseudo_cache[text] = (label, weight)
                for text, label, weight in result.self_play_examples:
                    synthetic_seen.add(text)
                    existing = pseudo_cache.get(text)
                    if existing is None or existing[1] < weight:
                        pseudo_cache[text] = (label, weight)
            for text, (label, weight) in pseudo_cache.items():
                final_texts.append(text)
                final_labels.append(label)
                final_weights.append(weight)
                if (
                    emotion_enabled
                    and emotion_lexicon is not None
                    and final_emotion_memory is not None
                    and emotion_dim > 0
                ):
                    vector = list(emotion_lexicon.vectorise(text))
                    final_emotion_vectors.append(vector)
                    label_idx = label_to_idx.get(label)
                    if label_idx is not None:
                        final_emotion_memory.register_vectors(
                            [label_idx],
                            [vector],
                            weights=[weight],
                        )
                elif (
                    emotion_enabled
                    and final_emotion_memory is not None
                    and emotion_dim > 0
                ):
                    final_emotion_vectors.append([0.0] * emotion_dim)
            pseudo_total = len(pseudo_seen)
            synthetic_total = len(synthetic_seen)
        print(
            f"Final full-data training: {len(final_texts)} supervised examples "
            f"({len(texts)} labelled + {pseudo_total} pseudo-labelled + {synthetic_total} synthetic self-play)."
        )

        final_curriculum_manager = None
        if args.adaptive_curriculum:
            final_curriculum_manager = AdaptiveCurriculum(
                start_epoch=args.curriculum_start_epoch,
                momentum=args.curriculum_momentum,
                min_multiplier=args.curriculum_min_multiplier,
                max_multiplier=args.curriculum_max_multiplier,
                hard_boost=args.curriculum_hard_boost,
                difficulty_power=args.curriculum_difficulty_power,
            )
            final_curriculum_manager.register_samples(final_texts, final_weights)

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
                    emotion_vectors=final_emotion_vectors if (emotion_enabled and emotion_dim > 0) else None,
                    emotion_dim=emotion_dim if emotion_enabled else 0,
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

        if emotion_enabled and emotion_lexicon is not None and emotion_dim > 0:
            augmented_emotion_vectors = list(final_emotion_vectors)
            base_len = len(final_emotion_vectors)
            if len(augmented_texts) > base_len:
                for new_text in augmented_texts[base_len:]:
                    augmented_emotion_vectors.append(list(emotion_lexicon.vectorise(new_text)))
        else:
            augmented_emotion_vectors = None

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
            emotion_vectors=augmented_emotion_vectors,
            emotion_dim=emotion_dim if emotion_enabled else 0,
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
                emotion_vectors=final_emotion_vectors if (emotion_enabled and emotion_dim > 0) else None,
                emotion_dim=emotion_dim if emotion_enabled else 0,
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
                    emotion_config=final_emotion_config,
                    meta_config=final_meta_config,
                    neuro_config=final_neuro_config,
                    discovery_config=final_discovery_config,
                    transcendent_config=final_transcendent_config,
                    frontier_config=final_frontier_config,
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

                eval_loss, eval_acc, eval_targets, eval_predictions, eval_probabilities = evaluate(
                    eval_model,
                    distill_eval_loader,
                    criterion,
                    device,
                    return_details=True,
                    emotion_config=final_emotion_config,
                )
                final_curriculum_summary: Optional[Dict[str, object]] = None
                if (
                    final_curriculum_manager is not None
                    and total_epochs >= args.curriculum_start_epoch
                ):
                    curriculum_weights = (
                        distillation_weights if distillation_weights is not None else final_weights
                    )
                    final_curriculum_summary = final_curriculum_manager.update_difficulties(
                        epoch=total_epochs,
                        stage="distill",
                        texts=final_texts,
                        labels=final_labels,
                        weights=curriculum_weights,
                        targets=eval_targets,
                        probabilities=eval_probabilities,
                        idx_to_label=idx_to_label_list,
                        snippet_fn=_truncate_snippet,
                    )
                    final_curriculum_manager.apply(final_texts, curriculum_weights)
                    if distillation_weights is not None:
                        final_curriculum_manager.apply(final_texts, final_weights)
                    base_limit = min(len(curriculum_weights), len(distill_dataset.examples))
                    for idx in range(base_limit):
                        distill_dataset.examples[idx].weight = float(curriculum_weights[idx])
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
                        "emotion_alignment": float(stats.get("emotion_alignment", 0.0)),
                        "meta_loss": float(stats.get("meta_loss", 0.0)),
                        "meta_attraction": float(stats.get("meta_attraction", 0.0)),
                        "meta_repulsion": float(stats.get("meta_repulsion", 0.0)),
                        "meta_novelty": float(stats.get("meta_novelty", 0.0)),
                        "meta_gap": float(stats.get("meta_gap", 0.0)),
                        "meta_entropy": float(stats.get("meta_entropy", 0.0)),
                        "meta_coverage": float(stats.get("meta_coverage", 0.0)),
                        "meta_updates": float(stats.get("meta_updates", 0.0)),
                        "neuro_loss": float(stats.get("neuro_loss", 0.0)),
                        "neuro_structural": float(stats.get("neuro_structural", 0.0)),
                        "neuro_semantic": float(stats.get("neuro_semantic", 0.0)),
                        "neuro_affective": float(stats.get("neuro_affective", 0.0)),
                        "neuro_entropy": float(stats.get("neuro_entropy", 0.0)),
                        "neuro_cohesion": float(stats.get("neuro_cohesion", 0.0)),
                        "neuro_updates": float(stats.get("neuro_updates", 0.0)),
                        "discovery_loss": float(stats.get("discovery_loss", 0.0)),
                        "discovery_alignment": float(stats.get("discovery_alignment", 0.0)),
                        "discovery_contrast": float(stats.get("discovery_contrast", 0.0)),
                        "discovery_imagination": float(stats.get("discovery_imagination", 0.0)),
                        "discovery_emotion": float(stats.get("discovery_emotion", 0.0)),
                        "discovery_confidence": float(stats.get("discovery_confidence", 0.0)),
                        "discovery_curiosity": float(stats.get("discovery_curiosity", 0.0)),
                        "discovery_counter_share": float(stats.get("discovery_counter_share", 0.0)),
                        "discovery_updates": float(stats.get("discovery_updates", 0.0)),
                        "transcendent_loss": float(stats.get("transcendent_loss", 0.0)),
                        "transcendent_stability": float(stats.get("transcendent_stability", 0.0)),
                        "transcendent_divergence": float(stats.get("transcendent_divergence", 0.0)),
                        "transcendent_foresight": float(stats.get("transcendent_foresight", 0.0)),
                        "transcendent_synthesis": float(stats.get("transcendent_synthesis", 0.0)),
                        "transcendent_affective": float(stats.get("transcendent_affective", 0.0)),
                        "transcendent_entropy": float(stats.get("transcendent_entropy", 0.0)),
                        "transcendent_coherence": float(stats.get("transcendent_coherence", 0.0)),
                        "transcendent_updates": float(stats.get("transcendent_updates", 0.0)),
                        "frontier_loss": float(stats.get("frontier_loss", 0.0)),
                        "frontier_novelty": float(stats.get("frontier_novelty", 0.0)),
                        "frontier_abstraction": float(stats.get("frontier_abstraction", 0.0)),
                        "frontier_transfer": float(stats.get("frontier_transfer", 0.0)),
                        "frontier_curiosity": float(stats.get("frontier_curiosity", 0.0)),
                        "frontier_emotion": float(stats.get("frontier_emotion", 0.0)),
                        "frontier_meta": float(stats.get("frontier_meta", 0.0)),
                        "frontier_diversity": float(stats.get("frontier_diversity", 0.0)),
                        "frontier_updates": float(stats.get("frontier_updates", 0.0)),
                    }
                )
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"Final stage distill epoch {total_epochs:03d} "
                    f"train_loss={train_loss:.4f} val_loss={eval_loss:.4f} "
                    f"train_acc={train_acc * 100:.2f}% val_acc={eval_acc * 100:.2f}% "
                    f"lr={current_lr:.6f} ({elapsed:.1f}s)"
                )
                if final_curriculum_summary:
                    hardest_examples = final_curriculum_summary.get("hardest_examples", [])
                    preview = "; ".join(
                        f"{item['label']}@{item['confidence']:.2f}→x{item['multiplier']:.2f}::{item['text']}"
                        for item in hardest_examples
                        if isinstance(item, dict)
                    )
                    if not preview:
                        preview = "n/a"
                    print(
                        f"   ↳ curriculum avg×{final_curriculum_summary['avg_multiplier']:.2f} "
                        f"(boosted {final_curriculum_summary['boosted']}, dampened {final_curriculum_summary['dampened']}, "
                        f"examples {final_curriculum_summary['examples']}); hardest {preview}"
                    )
                if final_meta_config is not None and final_meta_config.enabled:
                    print(
                        "   ↳ meta-introspection loss "
                        f"{stats.get('meta_loss', 0.0):.4f} "
                        f"gap {stats.get('meta_gap', 0.0):.3f} "
                        f"coverage {stats.get('meta_coverage', 0.0):.2f}"
                    )
                if final_neuro_config is not None and final_neuro_config.enabled:
                    print(
                        "   ↳ neuro-symbolic loss "
                        f"{stats.get('neuro_loss', 0.0):.4f} "
                        f"struct {stats.get('neuro_structural', 0.0):.4f} "
                        f"cohesion {stats.get('neuro_cohesion', 0.0):.3f} "
                        f"entropy {stats.get('neuro_entropy', 0.0):.3f}"
                    )
                if final_discovery_config is not None and final_discovery_config.enabled:
                    print(
                        "   ↳ self-discovery loss "
                        f"{stats.get('discovery_loss', 0.0):.4f} "
                        f"align {stats.get('discovery_alignment', 0.0):.4f} "
                        f"curiosity {stats.get('discovery_curiosity', 0.0):.3f}"
                    )
                if final_transcendent_config is not None and final_transcendent_config.enabled:
                    print(
                        "   ↳ transcendent cognition loss "
                        f"{stats.get('transcendent_loss', 0.0):.4f} "
                        f"coherence {stats.get('transcendent_coherence', 0.0):.3f} "
                        f"stability {stats.get('transcendent_stability', 0.0):.4f}"
                    )
                if final_frontier_config is not None and final_frontier_config.enabled:
                    print(
                        "   ↳ frontier intelligence loss "
                        f"{stats.get('frontier_loss', 0.0):.4f} "
                        f"novelty {stats.get('frontier_novelty', 0.0):.4f} "
                        f"diversity {stats.get('frontier_diversity', 0.0):.3f}"
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
                emotion_config=final_emotion_config,
                meta_config=final_meta_config,
                neuro_config=final_neuro_config,
                discovery_config=final_discovery_config,
                transcendent_config=final_transcendent_config,
                frontier_config=final_frontier_config,
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

            eval_loss, eval_acc, eval_targets, eval_predictions, eval_probabilities = evaluate(
                eval_model,
                eval_loader,
                criterion,
                device,
                return_details=True,
                emotion_config=final_emotion_config,
            )
            eval_metrics = compute_classification_metrics(
                eval_targets,
                eval_predictions,
                label_to_idx=label_to_idx,
            )
            current_lr = optimizer.param_groups[0]["lr"]
            final_curriculum_summary: Optional[Dict[str, object]] = None
            if (
                final_curriculum_manager is not None
                and total_epochs >= args.curriculum_start_epoch
            ):
                final_curriculum_summary = final_curriculum_manager.update_difficulties(
                    epoch=total_epochs,
                    stage="final_full",
                    texts=final_texts,
                    labels=final_labels,
                    weights=final_weights,
                    targets=eval_targets,
                    probabilities=eval_probabilities,
                    idx_to_label=idx_to_label_list,
                    snippet_fn=_truncate_snippet,
                )
                final_curriculum_manager.apply(final_texts, final_weights)
                base_limit = min(len(final_texts), len(train_dataset.examples))
                for idx in range(base_limit):
                    train_dataset.examples[idx].weight = float(final_weights[idx])
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
                    "emotion_alignment": float(stats.get("emotion_alignment", 0.0)),
                    "meta_loss": float(stats.get("meta_loss", 0.0)),
                    "meta_attraction": float(stats.get("meta_attraction", 0.0)),
                    "meta_repulsion": float(stats.get("meta_repulsion", 0.0)),
                    "meta_novelty": float(stats.get("meta_novelty", 0.0)),
                    "meta_gap": float(stats.get("meta_gap", 0.0)),
                    "meta_entropy": float(stats.get("meta_entropy", 0.0)),
                    "meta_coverage": float(stats.get("meta_coverage", 0.0)),
                    "meta_updates": float(stats.get("meta_updates", 0.0)),
                    "neuro_loss": float(stats.get("neuro_loss", 0.0)),
                    "neuro_structural": float(stats.get("neuro_structural", 0.0)),
                    "neuro_semantic": float(stats.get("neuro_semantic", 0.0)),
                    "neuro_affective": float(stats.get("neuro_affective", 0.0)),
                    "neuro_entropy": float(stats.get("neuro_entropy", 0.0)),
                    "neuro_cohesion": float(stats.get("neuro_cohesion", 0.0)),
                    "neuro_updates": float(stats.get("neuro_updates", 0.0)),
                    "discovery_loss": float(stats.get("discovery_loss", 0.0)),
                    "discovery_alignment": float(stats.get("discovery_alignment", 0.0)),
                    "discovery_contrast": float(stats.get("discovery_contrast", 0.0)),
                    "discovery_imagination": float(stats.get("discovery_imagination", 0.0)),
                    "discovery_emotion": float(stats.get("discovery_emotion", 0.0)),
                    "discovery_confidence": float(stats.get("discovery_confidence", 0.0)),
                    "discovery_curiosity": float(stats.get("discovery_curiosity", 0.0)),
                    "discovery_counter_share": float(stats.get("discovery_counter_share", 0.0)),
                    "discovery_updates": float(stats.get("discovery_updates", 0.0)),
                    "transcendent_loss": float(stats.get("transcendent_loss", 0.0)),
                    "transcendent_stability": float(stats.get("transcendent_stability", 0.0)),
                    "transcendent_divergence": float(stats.get("transcendent_divergence", 0.0)),
                    "transcendent_foresight": float(stats.get("transcendent_foresight", 0.0)),
                    "transcendent_synthesis": float(stats.get("transcendent_synthesis", 0.0)),
                    "transcendent_affective": float(stats.get("transcendent_affective", 0.0)),
                    "transcendent_entropy": float(stats.get("transcendent_entropy", 0.0)),
                    "transcendent_coherence": float(stats.get("transcendent_coherence", 0.0)),
                    "transcendent_updates": float(stats.get("transcendent_updates", 0.0)),
                    "frontier_loss": float(stats.get("frontier_loss", 0.0)),
                    "frontier_novelty": float(stats.get("frontier_novelty", 0.0)),
                    "frontier_abstraction": float(stats.get("frontier_abstraction", 0.0)),
                    "frontier_transfer": float(stats.get("frontier_transfer", 0.0)),
                    "frontier_curiosity": float(stats.get("frontier_curiosity", 0.0)),
                    "frontier_emotion": float(stats.get("frontier_emotion", 0.0)),
                    "frontier_meta": float(stats.get("frontier_meta", 0.0)),
                    "frontier_diversity": float(stats.get("frontier_diversity", 0.0)),
                    "frontier_updates": float(stats.get("frontier_updates", 0.0)),
                }
            )
            elapsed = time.perf_counter() - epoch_start
            print(
                f"Final training epoch {total_epochs:03d} "
                f"train_loss={train_loss:.4f} val_loss={eval_loss:.4f} "
                f"train_acc={train_acc * 100:.2f}% val_acc={eval_acc * 100:.2f}% "
                f"lr={current_lr:.6f} ({elapsed:.1f}s)"
            )
            if final_curriculum_summary:
                hardest_examples = final_curriculum_summary.get("hardest_examples", [])
                preview = "; ".join(
                    f"{item['label']}@{item['confidence']:.2f}→x{item['multiplier']:.2f}::{item['text']}"
                    for item in hardest_examples
                    if isinstance(item, dict)
                )
                if not preview:
                    preview = "n/a"
                print(
                    f"   ↳ curriculum avg×{final_curriculum_summary['avg_multiplier']:.2f} "
                    f"(boosted {final_curriculum_summary['boosted']}, dampened {final_curriculum_summary['dampened']}, "
                    f"examples {final_curriculum_summary['examples']}); hardest {preview}"
                )
            if final_meta_config is not None and final_meta_config.enabled:
                print(
                    "   ↳ meta-introspection loss "
                    f"{stats.get('meta_loss', 0.0):.4f} "
                    f"gap {stats.get('meta_gap', 0.0):.3f} "
                    f"coverage {stats.get('meta_coverage', 0.0):.2f}"
                )
            if final_neuro_config is not None and final_neuro_config.enabled:
                print(
                    "   ↳ neuro-symbolic loss "
                    f"{stats.get('neuro_loss', 0.0):.4f} "
                    f"struct {stats.get('neuro_structural', 0.0):.4f} "
                    f"cohesion {stats.get('neuro_cohesion', 0.0):.3f} "
                    f"entropy {stats.get('neuro_entropy', 0.0):.3f}"
                )
            if final_discovery_config is not None and final_discovery_config.enabled:
                print(
                    "   ↳ self-discovery loss "
                    f"{stats.get('discovery_loss', 0.0):.4f} "
                    f"align {stats.get('discovery_alignment', 0.0):.4f} "
                    f"curiosity {stats.get('discovery_curiosity', 0.0):.3f}"
                )
            if final_transcendent_config is not None and final_transcendent_config.enabled:
                print(
                    "   ↳ transcendent cognition loss "
                    f"{stats.get('transcendent_loss', 0.0):.4f} "
                    f"coherence {stats.get('transcendent_coherence', 0.0):.3f} "
                    f"stability {stats.get('transcendent_stability', 0.0):.4f}"
                )
            if final_frontier_config is not None and final_frontier_config.enabled:
                print(
                    "   ↳ frontier intelligence loss "
                    f"{stats.get('frontier_loss', 0.0):.4f} "
                    f"novelty {stats.get('frontier_novelty', 0.0):.4f} "
                    f"diversity {stats.get('frontier_diversity', 0.0):.3f}"
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
            emotion_config=final_emotion_config,
        )
        final_metrics = compute_classification_metrics(
            final_targets,
            final_predictions,
            label_to_idx=label_to_idx,
        )

        final_meta_snapshot: Optional[Dict[str, object]] = None
        if final_meta_config is not None and final_meta_config.enabled:
            final_meta_snapshot = final_meta_config.introspector.snapshot()

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
            "synthetic_examples_used": synthetic_total,
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
            "adaptive_curriculum": (
                final_curriculum_manager.export_metadata()
                if final_curriculum_manager is not None
                else {"enabled": bool(args.adaptive_curriculum)}
            ),
            "emotion_reasoner": (
                {
                    "enabled": True,
                    "dimension": emotion_dim,
                    "consistency_weight": args.emotion_consistency_weight,
                    "temperature": args.emotion_expectation_temperature,
                    "prototype_smoothing": args.emotion_prototype_smoothing,
                    "fusion_dropout": args.emotion_fusion_dropout,
                    "memory_updates": int(final_emotion_memory.total_updates),
                }
                if emotion_enabled and final_emotion_memory is not None and emotion_dim > 0
                else {"enabled": bool(emotion_enabled)}
            ),
            "meta_introspection": (
                {
                    "enabled": True,
                    "attraction_weight": final_meta_config.attraction_weight,
                    "repulsion_weight": final_meta_config.repulsion_weight,
                    "discovery_weight": final_meta_config.discovery_weight,
                    "gap_margin": final_meta_config.gap_margin,
                    "temperature": final_meta_config.temperature,
                    "momentum": final_meta_config.introspector.momentum,
                    "margin": final_meta_config.introspector.margin,
                    "history_limit": final_meta_config.introspector.history.maxlen,
                    "snapshot": final_meta_snapshot,
                }
                if final_meta_config is not None and final_meta_config.enabled
                else {"enabled": False}
            ),
            "neuro_symbolic": (
                {
                    "enabled": True,
                    "structural_weight": final_neuro_config.structural_weight,
                    "semantic_weight": final_neuro_config.semantic_weight,
                    "affective_weight": final_neuro_config.affective_weight,
                    "temperature": final_neuro_config.temperature,
                    "self_loop": final_neuro_config.self_loop,
                    "lexical_weight": final_neuro_config.reasoner.lexical_weight,
                    "graph_momentum": final_neuro_config.reasoner.graph_momentum,
                    "feature_momentum": final_neuro_config.reasoner.feature_momentum,
                    "min_confidence": final_neuro_config.reasoner.min_confidence,
                    "history_limit": final_neuro_config.reasoner.history.maxlen,
                    "snapshot": final_neuro_config.reasoner.snapshot(),
                    "metadata": final_neuro_config.reasoner.export_metadata(),
                }
                if final_neuro_config is not None and final_neuro_config.enabled
                else {"enabled": bool(args.neuro_symbolic_reasoner)}
            ),
            "self_discovery": (
                {
                    "enabled": True,
                    "alignment_weight": final_discovery_config.alignment_weight,
                    "contrast_weight": final_discovery_config.contrast_weight,
                    "imagination_weight": final_discovery_config.imagination_weight,
                    "emotion_weight": final_discovery_config.emotion_weight,
                    "temperature": final_discovery_config.temperature,
                    "min_confidence": final_discovery_config.min_confidence,
                    "margin": final_discovery_config.margin,
                    "feature_momentum": final_discovery_config.orchestrator.feature_momentum,
                    "counter_momentum": final_discovery_config.orchestrator.counter_momentum,
                    "imagination_momentum": final_discovery_config.orchestrator.imagination_momentum,
                    "curiosity_weight": final_discovery_config.orchestrator.curiosity_weight,
                    "history_limit": final_discovery_config.orchestrator.history.maxlen,
                    "snapshot": final_discovery_config.orchestrator.snapshot(),
                    "metadata": final_discovery_config.orchestrator.export_metadata(),
                }
                if final_discovery_config is not None and final_discovery_config.enabled
                else {"enabled": bool(args.self_discovery)}
            ),
            "transcendent_cognition": (
                {
                    "enabled": True,
                    "stability_weight": final_transcendent_config.stability_weight,
                    "divergence_weight": final_transcendent_config.divergence_weight,
                    "foresight_weight": final_transcendent_config.foresight_weight,
                    "synthesis_weight": final_transcendent_config.synthesis_weight,
                    "affective_weight": final_transcendent_config.affective_weight,
                    "entropy_weight": final_transcendent_config.entropy_weight,
                    "temperature": final_transcendent_config.temperature,
                    "margin": final_transcendent_config.margin,
                    "feature_momentum": final_transcendent_config.architect.feature_momentum,
                    "counter_momentum": final_transcendent_config.architect.counter_momentum,
                    "transition_momentum": final_transcendent_config.architect.transition_momentum,
                    "imagination_momentum": final_transcendent_config.architect.imagination_momentum,
                    "max_glimpses": final_transcendent_config.architect.max_glimpses,
                    "history_limit": final_transcendent_config.architect.history.maxlen,
                    "snapshot": final_transcendent_config.architect.snapshot(),
                    "metadata": final_transcendent_config.architect.export_metadata(),
                }
                if final_transcendent_config is not None and final_transcendent_config.enabled
                else {"enabled": bool(args.transcendent_cognition)}
            ),
            "frontier_intelligence": (
                {
                    "enabled": True,
                    "novelty_weight": final_frontier_config.novelty_weight,
                    "abstraction_weight": final_frontier_config.abstraction_weight,
                    "transfer_weight": final_frontier_config.transfer_weight,
                    "curiosity_weight": final_frontier_config.curiosity_weight,
                    "emotion_weight": final_frontier_config.emotion_weight,
                    "meta_weight": final_frontier_config.meta_weight,
                    "temperature": final_frontier_config.temperature,
                    "margin": final_frontier_config.margin,
                    "concept_momentum": final_frontier_config.catalyst.concept_momentum,
                    "bridge_momentum": final_frontier_config.catalyst.bridge_momentum,
                    "novelty_momentum": final_frontier_config.catalyst.novelty_momentum,
                    "meta_momentum": final_frontier_config.catalyst.meta_momentum,
                    "emotion_momentum": final_frontier_config.catalyst.emotion_momentum,
                    "history_limit": final_frontier_config.catalyst.history.maxlen,
                    "snapshot": final_frontier_config.catalyst.snapshot(),
                    "metadata": final_frontier_config.catalyst.export_metadata(),
                }
                if final_frontier_config is not None and final_frontier_config.enabled
                else {"enabled": bool(args.frontier_intelligence)}
            ),
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

        final_emotion_alignment_values = [
            float(entry.get("emotion_alignment", 0.0))
            for entry in history
            if "emotion_alignment" in entry
        ]
        final_meta_loss_values = [
            float(entry.get("meta_loss", 0.0))
            for entry in history
            if "meta_loss" in entry
        ]
        final_meta_gap_values = [
            float(entry.get("meta_gap", 0.0))
            for entry in history
            if "meta_gap" in entry
        ]
        final_meta_entropy_values = [
            float(entry.get("meta_entropy", 0.0))
            for entry in history
            if "meta_entropy" in entry
        ]
        final_meta_coverage_values = [
            float(entry.get("meta_coverage", 0.0))
            for entry in history
            if "meta_coverage" in entry
        ]
        final_meta_update_values = [
            float(entry.get("meta_updates", 0.0))
            for entry in history
            if "meta_updates" in entry
        ]
        final_discovery_loss_values = [
            float(entry.get("discovery_loss", 0.0))
            for entry in history
            if "discovery_loss" in entry
        ]
        final_discovery_alignment_values = [
            float(entry.get("discovery_alignment", 0.0))
            for entry in history
            if "discovery_alignment" in entry
        ]
        final_discovery_contrast_values = [
            float(entry.get("discovery_contrast", 0.0))
            for entry in history
            if "discovery_contrast" in entry
        ]
        final_discovery_imagination_values = [
            float(entry.get("discovery_imagination", 0.0))
            for entry in history
            if "discovery_imagination" in entry
        ]
        final_discovery_emotion_values = [
            float(entry.get("discovery_emotion", 0.0))
            for entry in history
            if "discovery_emotion" in entry
        ]
        final_discovery_confidence_values = [
            float(entry.get("discovery_confidence", 0.0))
            for entry in history
            if "discovery_confidence" in entry
        ]
        final_discovery_curiosity_values = [
            float(entry.get("discovery_curiosity", 0.0))
            for entry in history
            if "discovery_curiosity" in entry
        ]
        final_discovery_counter_values = [
            float(entry.get("discovery_counter_share", 0.0))
            for entry in history
            if "discovery_counter_share" in entry
        ]
        final_discovery_update_values = [
            float(entry.get("discovery_updates", 0.0))
            for entry in history
            if "discovery_updates" in entry
        ]
        final_transcendent_loss_values = [
            float(entry.get("transcendent_loss", 0.0))
            for entry in history
            if "transcendent_loss" in entry
        ]
        final_transcendent_stability_values = [
            float(entry.get("transcendent_stability", 0.0))
            for entry in history
            if "transcendent_stability" in entry
        ]
        final_transcendent_divergence_values = [
            float(entry.get("transcendent_divergence", 0.0))
            for entry in history
            if "transcendent_divergence" in entry
        ]
        final_transcendent_foresight_values = [
            float(entry.get("transcendent_foresight", 0.0))
            for entry in history
            if "transcendent_foresight" in entry
        ]
        final_transcendent_synthesis_values = [
            float(entry.get("transcendent_synthesis", 0.0))
            for entry in history
            if "transcendent_synthesis" in entry
        ]
        final_transcendent_affective_values = [
            float(entry.get("transcendent_affective", 0.0))
            for entry in history
            if "transcendent_affective" in entry
        ]
        final_transcendent_entropy_values = [
            float(entry.get("transcendent_entropy", 0.0))
            for entry in history
            if "transcendent_entropy" in entry
        ]
        final_transcendent_coherence_values = [
            float(entry.get("transcendent_coherence", 0.0))
            for entry in history
            if "transcendent_coherence" in entry
        ]
        final_transcendent_update_values = [
            float(entry.get("transcendent_updates", 0.0))
            for entry in history
            if "transcendent_updates" in entry
        ]
        final_frontier_loss_values = [
            float(entry.get("frontier_loss", 0.0))
            for entry in history
            if "frontier_loss" in entry
        ]
        final_frontier_novelty_values = [
            float(entry.get("frontier_novelty", 0.0))
            for entry in history
            if "frontier_novelty" in entry
        ]
        final_frontier_abstraction_values = [
            float(entry.get("frontier_abstraction", 0.0))
            for entry in history
            if "frontier_abstraction" in entry
        ]
        final_frontier_transfer_values = [
            float(entry.get("frontier_transfer", 0.0))
            for entry in history
            if "frontier_transfer" in entry
        ]
        final_frontier_curiosity_values = [
            float(entry.get("frontier_curiosity", 0.0))
            for entry in history
            if "frontier_curiosity" in entry
        ]
        final_frontier_emotion_values = [
            float(entry.get("frontier_emotion", 0.0))
            for entry in history
            if "frontier_emotion" in entry
        ]
        final_frontier_meta_values = [
            float(entry.get("frontier_meta", 0.0))
            for entry in history
            if "frontier_meta" in entry
        ]
        final_frontier_diversity_values = [
            float(entry.get("frontier_diversity", 0.0))
            for entry in history
            if "frontier_diversity" in entry
        ]
        final_frontier_update_values = [
            float(entry.get("frontier_updates", 0.0))
            for entry in history
            if "frontier_updates" in entry
        ]
        final_neuro_loss_values = [
            float(entry.get("neuro_loss", 0.0))
            for entry in history
            if "neuro_loss" in entry
        ]
        final_neuro_struct_values = [
            float(entry.get("neuro_structural", 0.0))
            for entry in history
            if "neuro_structural" in entry
        ]
        final_neuro_semantic_values = [
            float(entry.get("neuro_semantic", 0.0))
            for entry in history
            if "neuro_semantic" in entry
        ]
        final_neuro_affective_values = [
            float(entry.get("neuro_affective", 0.0))
            for entry in history
            if "neuro_affective" in entry
        ]
        final_neuro_entropy_values = [
            float(entry.get("neuro_entropy", 0.0))
            for entry in history
            if "neuro_entropy" in entry
        ]
        final_neuro_cohesion_values = [
            float(entry.get("neuro_cohesion", 0.0))
            for entry in history
            if "neuro_cohesion" in entry
        ]
        final_neuro_update_values = [
            float(entry.get("neuro_updates", 0.0))
            for entry in history
            if "neuro_updates" in entry
        ]
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
            "synthetic_examples_used": synthetic_total,
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
        metrics["emotion_reasoner_enabled"] = bool(emotion_enabled and emotion_dim > 0)
        if emotion_enabled and final_emotion_memory is not None and emotion_dim > 0:
            metrics["emotion_memory_updates"] = int(final_emotion_memory.total_updates)
        if emotion_enabled and final_emotion_alignment_values:
            metrics["emotion_alignment_mean"] = float(
                sum(final_emotion_alignment_values) / len(final_emotion_alignment_values)
            )
            metrics["emotion_alignment_last"] = float(final_emotion_alignment_values[-1])
        metrics["meta_introspection_enabled"] = bool(final_meta_config is not None and final_meta_config.enabled)
        if final_meta_loss_values:
            metrics["meta_loss_mean"] = float(
                sum(final_meta_loss_values) / len(final_meta_loss_values)
            )
            metrics["meta_loss_last"] = float(final_meta_loss_values[-1])
        if final_meta_gap_values:
            metrics["meta_gap_mean"] = float(
                sum(final_meta_gap_values) / len(final_meta_gap_values)
            )
            metrics["meta_gap_last"] = float(final_meta_gap_values[-1])
        if final_meta_entropy_values:
            metrics["meta_entropy_mean"] = float(
                sum(final_meta_entropy_values) / len(final_meta_entropy_values)
            )
            metrics["meta_entropy_last"] = float(final_meta_entropy_values[-1])
        if final_meta_coverage_values:
            metrics["meta_coverage_mean"] = float(
                sum(final_meta_coverage_values) / len(final_meta_coverage_values)
            )
            metrics["meta_coverage_last"] = float(final_meta_coverage_values[-1])
        if final_meta_update_values:
            metrics["meta_updates_last"] = float(final_meta_update_values[-1])
        metrics["self_discovery_enabled"] = bool(
            final_discovery_config is not None and final_discovery_config.enabled
        )
        if final_discovery_loss_values:
            metrics["discovery_loss_mean"] = float(
                sum(final_discovery_loss_values) / len(final_discovery_loss_values)
            )
            metrics["discovery_loss_last"] = float(final_discovery_loss_values[-1])
        if final_discovery_alignment_values:
            metrics["discovery_alignment_mean"] = float(
                sum(final_discovery_alignment_values) / len(final_discovery_alignment_values)
            )
            metrics["discovery_alignment_last"] = float(final_discovery_alignment_values[-1])
        if final_discovery_contrast_values:
            metrics["discovery_contrast_mean"] = float(
                sum(final_discovery_contrast_values) / len(final_discovery_contrast_values)
            )
            metrics["discovery_contrast_last"] = float(final_discovery_contrast_values[-1])
        if final_discovery_imagination_values:
            metrics["discovery_imagination_mean"] = float(
                sum(final_discovery_imagination_values) / len(final_discovery_imagination_values)
            )
            metrics["discovery_imagination_last"] = float(final_discovery_imagination_values[-1])
        if final_discovery_emotion_values:
            metrics["discovery_emotion_mean"] = float(
                sum(final_discovery_emotion_values) / len(final_discovery_emotion_values)
            )
            metrics["discovery_emotion_last"] = float(final_discovery_emotion_values[-1])
        if final_discovery_confidence_values:
            metrics["discovery_confidence_mean"] = float(
                sum(final_discovery_confidence_values) / len(final_discovery_confidence_values)
            )
            metrics["discovery_confidence_last"] = float(final_discovery_confidence_values[-1])
        if final_discovery_curiosity_values:
            metrics["discovery_curiosity_mean"] = float(
                sum(final_discovery_curiosity_values) / len(final_discovery_curiosity_values)
            )
            metrics["discovery_curiosity_last"] = float(final_discovery_curiosity_values[-1])
        if final_discovery_counter_values:
            metrics["discovery_counter_share_mean"] = float(
                sum(final_discovery_counter_values) / len(final_discovery_counter_values)
            )
            metrics["discovery_counter_share_last"] = float(final_discovery_counter_values[-1])
        if final_discovery_update_values:
            metrics["discovery_updates_last"] = float(final_discovery_update_values[-1])
        metrics["transcendent_cognition_enabled"] = bool(
            final_transcendent_config is not None and final_transcendent_config.enabled
        )
        if final_transcendent_loss_values:
            metrics["transcendent_loss_mean"] = float(
                sum(final_transcendent_loss_values) / len(final_transcendent_loss_values)
            )
            metrics["transcendent_loss_last"] = float(final_transcendent_loss_values[-1])
        if final_transcendent_stability_values:
            metrics["transcendent_stability_mean"] = float(
                sum(final_transcendent_stability_values) / len(final_transcendent_stability_values)
            )
            metrics["transcendent_stability_last"] = float(final_transcendent_stability_values[-1])
        if final_transcendent_divergence_values:
            metrics["transcendent_divergence_mean"] = float(
                sum(final_transcendent_divergence_values) / len(final_transcendent_divergence_values)
            )
            metrics["transcendent_divergence_last"] = float(final_transcendent_divergence_values[-1])
        if final_transcendent_foresight_values:
            metrics["transcendent_foresight_mean"] = float(
                sum(final_transcendent_foresight_values) / len(final_transcendent_foresight_values)
            )
            metrics["transcendent_foresight_last"] = float(final_transcendent_foresight_values[-1])
        if final_transcendent_synthesis_values:
            metrics["transcendent_synthesis_mean"] = float(
                sum(final_transcendent_synthesis_values) / len(final_transcendent_synthesis_values)
            )
            metrics["transcendent_synthesis_last"] = float(final_transcendent_synthesis_values[-1])
        if final_transcendent_affective_values:
            metrics["transcendent_affective_mean"] = float(
                sum(final_transcendent_affective_values) / len(final_transcendent_affective_values)
            )
            metrics["transcendent_affective_last"] = float(final_transcendent_affective_values[-1])
        if final_transcendent_entropy_values:
            metrics["transcendent_entropy_mean"] = float(
                sum(final_transcendent_entropy_values) / len(final_transcendent_entropy_values)
            )
            metrics["transcendent_entropy_last"] = float(final_transcendent_entropy_values[-1])
        if final_transcendent_coherence_values:
            metrics["transcendent_coherence_mean"] = float(
                sum(final_transcendent_coherence_values) / len(final_transcendent_coherence_values)
            )
            metrics["transcendent_coherence_last"] = float(final_transcendent_coherence_values[-1])
        if final_transcendent_update_values:
            metrics["transcendent_updates_last"] = float(final_transcendent_update_values[-1])
        metrics["frontier_intelligence_enabled"] = bool(
            final_frontier_config is not None and final_frontier_config.enabled
        )
        if final_frontier_loss_values:
            metrics["frontier_loss_mean"] = float(
                sum(final_frontier_loss_values) / len(final_frontier_loss_values)
            )
            metrics["frontier_loss_last"] = float(final_frontier_loss_values[-1])
        if final_frontier_novelty_values:
            metrics["frontier_novelty_mean"] = float(
                sum(final_frontier_novelty_values) / len(final_frontier_novelty_values)
            )
            metrics["frontier_novelty_last"] = float(final_frontier_novelty_values[-1])
        if final_frontier_abstraction_values:
            metrics["frontier_abstraction_mean"] = float(
                sum(final_frontier_abstraction_values) / len(final_frontier_abstraction_values)
            )
            metrics["frontier_abstraction_last"] = float(final_frontier_abstraction_values[-1])
        if final_frontier_transfer_values:
            metrics["frontier_transfer_mean"] = float(
                sum(final_frontier_transfer_values) / len(final_frontier_transfer_values)
            )
            metrics["frontier_transfer_last"] = float(final_frontier_transfer_values[-1])
        if final_frontier_curiosity_values:
            metrics["frontier_curiosity_mean"] = float(
                sum(final_frontier_curiosity_values) / len(final_frontier_curiosity_values)
            )
            metrics["frontier_curiosity_last"] = float(final_frontier_curiosity_values[-1])
        if final_frontier_emotion_values:
            metrics["frontier_emotion_mean"] = float(
                sum(final_frontier_emotion_values) / len(final_frontier_emotion_values)
            )
            metrics["frontier_emotion_last"] = float(final_frontier_emotion_values[-1])
        if final_frontier_meta_values:
            metrics["frontier_meta_mean"] = float(
                sum(final_frontier_meta_values) / len(final_frontier_meta_values)
            )
            metrics["frontier_meta_last"] = float(final_frontier_meta_values[-1])
        if final_frontier_diversity_values:
            metrics["frontier_diversity_mean"] = float(
                sum(final_frontier_diversity_values) / len(final_frontier_diversity_values)
            )
            metrics["frontier_diversity_last"] = float(final_frontier_diversity_values[-1])
        if final_frontier_update_values:
            metrics["frontier_updates_last"] = float(final_frontier_update_values[-1])
        metrics["neuro_symbolic_enabled"] = bool(final_neuro_config is not None and final_neuro_config.enabled)
        if final_neuro_loss_values:
            metrics["neuro_loss_mean"] = float(sum(final_neuro_loss_values) / len(final_neuro_loss_values))
            metrics["neuro_loss_last"] = float(final_neuro_loss_values[-1])
        if final_neuro_struct_values:
            metrics["neuro_structural_mean"] = float(sum(final_neuro_struct_values) / len(final_neuro_struct_values))
            metrics["neuro_structural_last"] = float(final_neuro_struct_values[-1])
        if final_neuro_semantic_values:
            metrics["neuro_semantic_mean"] = float(sum(final_neuro_semantic_values) / len(final_neuro_semantic_values))
            metrics["neuro_semantic_last"] = float(final_neuro_semantic_values[-1])
        if final_neuro_affective_values:
            metrics["neuro_affective_mean"] = float(sum(final_neuro_affective_values) / len(final_neuro_affective_values))
            metrics["neuro_affective_last"] = float(final_neuro_affective_values[-1])
        if final_neuro_entropy_values:
            metrics["neuro_entropy_mean"] = float(sum(final_neuro_entropy_values) / len(final_neuro_entropy_values))
            metrics["neuro_entropy_last"] = float(final_neuro_entropy_values[-1])
        if final_neuro_cohesion_values:
            metrics["neuro_cohesion_mean"] = float(sum(final_neuro_cohesion_values) / len(final_neuro_cohesion_values))
            metrics["neuro_cohesion_last"] = float(final_neuro_cohesion_values[-1])
        if final_neuro_update_values:
            metrics["neuro_updates_last"] = float(final_neuro_update_values[-1])

        if args.encoder_type == "transformer":
            metrics["transformer_model"] = args.transformer_model
            if tokenizer_obj is not None:
                metrics["tokenizer_vocab_size"] = len(tokenizer_obj)
        if args.encoder_type == "st":
            metrics["sentence_transformer_model"] = args.sentence_transformer_model
            metrics["sentence_transformer_hidden_dim"] = args.st_hidden_dim
            metrics["sentence_transformer_dropout"] = args.st_dropout
        if final_curriculum_manager is not None:
            metrics.update(final_curriculum_manager.export_metrics())
        elif args.adaptive_curriculum:
            metrics.setdefault("curriculum_updates", 0)
            metrics.setdefault("curriculum_avg_multiplier", 1.0)
            metrics.setdefault("curriculum_max_multiplier", 1.0)
            metrics.setdefault("curriculum_min_multiplier", 1.0)

        evaluation_outputs: List[Dict[str, object]] = []
        for sample in evaluation_inputs:
            prediction = predict_with_trace(
                final_model,
                sample,
                vocab=vocab,
                label_to_idx=label_to_idx,
                max_len=max_seq_len,
                device=device,
                tokenizer=tokenizer_obj,
                tokenizer_cache=tokenizer_cache_fn,
                embedding_model=embedding_fn,
                emotion_encoder=emotion_lexicon if (emotion_enabled and emotion_dim > 0) else None,
                emotion_dim=emotion_dim,
                emotion_config=final_emotion_config,
            )
            response = generate_response(prediction.label, sample)
            entry: Dict[str, object] = {
                "input": sample,
                "predicted_intent": prediction.label,
                "confidence": prediction.confidence,
                "top_intents": [
                    {"label": label, "confidence": score}
                    for label, score in prediction.top_predictions
                ],
                "response": response.message,
                "response_strategy": response.strategy,
            }
            if response.basis:
                entry["response_basis"] = response.basis
            evaluation_outputs.append(entry)

        final_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "metadata": metadata,
            "metrics": metrics,
            "model_state": best_state,
            "evaluation_outputs": evaluation_outputs,
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
        final_metadata["evaluation_outputs"] = final_stage["evaluation_outputs"]
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

    print("\nEvaluation showcase (best fold model):")
    for entry in best_fold.evaluation_outputs:
        print(f"Input: {entry['input']}")
        confidence = entry.get("confidence")
        if confidence is not None:
            print(
                "  Predicted intent: "
                f"{entry['predicted_intent']} ({confidence:.2%} confidence)"
            )
        else:
            print(f"  Predicted intent: {entry['predicted_intent']}")
        top_candidates = entry.get("top_intents") or []
        if top_candidates:
            print("  Ranked intents:")
            for candidate in top_candidates:
                label = candidate.get("label")
                score = candidate.get("confidence")
                marker = " (selected)" if label == entry.get("predicted_intent") else ""
                if score is not None:
                    print(f"    {label}: {score:.2%}{marker}")
                else:
                    print(f"    {label}{marker}")
        strategy = entry.get("response_strategy")
        if strategy:
            print(f"  Response [{strategy}]: {entry['response']}")
        else:
            print(f"  Response: {entry['response']}")
        basis = entry.get("response_basis")
        if basis:
            print(f"    Basis: {basis}")
        print()


if __name__ == "__main__":
    main()
