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

Monte Carlo consensus checks score every pseudo-label across multiple
stochastic forward passes, filtering out uncertain candidates while scaling
their weights by agreement and variance so that only stable signals influence
the optimiser.

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

import atexit
import argparse
import csv
import json
import random
import re
import contextlib
import copy
import gc
import unicodedata
import fnmatch
import hashlib
import math
import platform
import posixpath
import os
import shutil
import time
import warnings
import importlib
import importlib.util
import subprocess
import sys
import threading
import string
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol, TYPE_CHECKING, TypeAlias, SupportsFloat, TypeGuard, TypeVar, cast
from collections.abc import Callable, Iterable, Mapping, MutableSequence, Sequence
from types import SimpleNamespace

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128"
)
try:  # NumPy is optional for verification-only runs but required for training paths.
    import numpy as np  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal environments
    class _NumpyMissingProxy:
        """Stand-in object that surfaces a clear error when NumPy is unavailable."""

        __slots__ = ("_message",)

        def __init__(self) -> None:
            self._message = (
                "NumPy is required for full intent classifier training. "
                "Install it with 'pip install numpy' to unlock all features."
            )

        def __getattr__(self, attribute: str):  # pragma: no cover - simple error surface
            raise ModuleNotFoundError(
                f"{self._message} (attempted to access numpy.{attribute})."
            )

        def __call__(self, *args: Any, **kwargs: Any):  # pragma: no cover - simple error surface
            raise ModuleNotFoundError(self._message)

        def __bool__(self) -> bool:  # pragma: no cover - keeps truthiness predictable
            return False

    np = _NumpyMissingProxy()  # type: ignore[assignment]
    numpy_available = False
else:
    numpy_available = True

if TYPE_CHECKING:
    from numpy import ndarray  # type: ignore[import-not-found]
else:
    try:
        from numpy import ndarray  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - exercised in minimal environments
        ndarray = Any  # type: ignore[assignment]

if not numpy_available:
    warnings.filterwarnings(
        "ignore",
        message=r"Failed to initialize NumPy: No module named 'numpy'",
        category=UserWarning,
        module=r"torch\._subclasses\.functional_tensor",
    )

_nvml_loaded = False


if TYPE_CHECKING:
    import torch as _torch_typing  # type: ignore[import-not-found]
    from torch import optim as _torch_optim  # type: ignore[import-not-found]
    from torch.optim.lr_scheduler import LRScheduler as _TorchLRScheduler  # type: ignore[import-not-found]

    TorchTensor: TypeAlias = _torch_typing.Tensor
    TorchDevice: TypeAlias = _torch_typing.device
    TorchDType: TypeAlias = _torch_typing.dtype
    TorchOptimizer: TypeAlias = _torch_optim.Optimizer
    SchedulerLike: TypeAlias = _TorchLRScheduler
    TorchRuntimeTensor: TypeAlias = _torch_typing.Tensor
    TorchTensorType = _torch_typing.Tensor
else:  # pragma: no cover - runtime-only shim when torch isn't importable
    TorchTensor: TypeAlias = Any
    TorchDevice: TypeAlias = Any
    TorchDType: TypeAlias = Any
    TorchOptimizer: TypeAlias = Any
    SchedulerLike: TypeAlias = Any
    TorchRuntimeTensor = Any
    TorchTensorType = Any


VectorLike = Sequence[float] | TorchTensor
TokenizerCacheFn: TypeAlias = Callable[[str], tuple[tuple[int, ...], tuple[int, ...]]]


BatchT = TypeVar("BatchT")
BatchItem: TypeAlias = TorchTensor | None
BatchExample: TypeAlias = tuple[BatchItem, ...]


class _ScalerProtocol(Protocol):
    def fit_transform(self, X: ndarray, y: ndarray | None = ...) -> ndarray: ...

    def transform(self, X: ndarray) -> ndarray: ...


class _LogisticModelProtocol(Protocol):
    def fit(self, X: ndarray, y: ndarray) -> Any: ...

    def predict(self, X: ndarray) -> ndarray: ...

    def decision_function(self, X: ndarray) -> ndarray: ...


class _ScalerFactory(Protocol):
    def __call__(self) -> _ScalerProtocol: ...


class _LogisticModelFactory(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> _LogisticModelProtocol: ...


class TokenizerProtocol(Protocol):
    """Minimal protocol covering the tokenizer interactions used in training."""

    def __call__(
        self,
        texts: Sequence[str],
        *,
        padding: str,
        truncation: bool,
        max_length: int,
        return_attention_mask: bool,
    ) -> Mapping[str, Any]:
        ...

    def __len__(self) -> int:
        ...


def _nvml_safe_shutdown() -> None:
    """Attempt to cleanly shut down NVML when the process exits."""

    if not _nvml_loaded:
        return
    if pynvml is None:  # pragma: no cover - defensive
        return
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass


try:
    import pynvml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore[assignment]
else:  # pragma: no cover - optional dependency
    try:
        pynvml.nvmlInit()
    except Exception:
        pynvml = None  # type: ignore[assignment]
    else:
        _nvml_loaded = True
        atexit.register(_nvml_safe_shutdown)
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean, median, pstdev
from collections.abc import Callable, Iterable, Mapping, MutableSequence

try:
    _this_file = Path(__file__).resolve()
except NameError:  # pragma: no cover - __file__ is undefined inside interactive shells
    _this_file = None


_SETUP_SENTINEL_ENV = "SOLARUS_TRAINER_SKIP_AUTO_SETUP"
_FORCE_SETUP_ENV = "SOLARUS_TRAINER_FORCE_SETUP"
_CUDA_BOOTSTRAP_SENTINEL_ENV = "SOLARUS_TRAINER_CUDA_BOOTSTRAP_ATTEMPTED"
_PREFERRED_CUDA_VARIANT_ENV = "SOLARUS_TRAINER_PREFERRED_CUDA_VARIANT"
_REQUIRE_CUDA_ENV = "SOLARUS_TRAINER_REQUIRE_CUDA"
_SETUP_BOOL_TRUE = {"1", "true", "yes", "on"}
_SETUP_BOOL_FALSE = {"0", "false", "no", "off"}
_OPTIONAL_INSTALL_ATTEMPTS: set[str] = set()
_CUDA_VARIANT_FALLBACKS: dict[str, tuple[str, ...]] = {
    "cu124": ("cu121", "cu118"),
    "cu121": ("cu118",),
    "cu118": (),
    "cu117": (),
    "cu116": (),
}
_EXPECT_CUDA_ENV = "SOLARUS_TRAINER_EXPECTS_CUDA"


def _env_flag_active(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _SETUP_BOOL_TRUE


def _resolve_project_root() -> Path:
    if _this_file is not None:
        return _this_file.parent
    return Path.cwd().resolve()


def _default_notes() -> dict[str, Any]:
    return {}


@dataclass
class SpeedTestComplexityProfile:
    """Configuration used to upscale per-pass timings during projections."""

    baseline_multiplier: float = 1.0
    override_pass_seconds: float | None = None
    constant_overhead_seconds: float = 0.0
    notes: Mapping[str, Any] = field(default_factory=_default_notes)

    def resolve_pass_seconds(self, baseline: float | None) -> float | None:
        if self.override_pass_seconds is not None and self.override_pass_seconds > 0:
            base = float(self.override_pass_seconds)
        elif baseline is not None and baseline > 0:
            base = float(baseline)
        else:
            return None
        multiplier = max(self.baseline_multiplier, 0.0)
        if multiplier == 0:
            return 0.0
        return max(0.0, base * multiplier)


@dataclass
class DatasetSummary:
    """Lightweight overview of a labelled dataset used for runtime projections."""

    path: str | None = None
    examples: int = 0
    empty_rows: int = 0
    scanned_rows: int = 0
    token_samples: int = 0
    sample_size: int = 0
    average_tokens: float = 0.0
    total_tokens: float = 0.0
    truncated: bool = False


@dataclass
class SpeedTestEstimateComponent:
    """Rich description of a per-pass estimate candidate used in blending."""

    source: str
    seconds: float
    weight: float
    confidence: float
    evidence: float
    adjusted_weight: float
    jitter: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "seconds": float(self.seconds),
            "weight": float(self.weight),
            "confidence": float(self.confidence),
            "evidence": float(self.evidence),
            "adjusted_weight": float(self.adjusted_weight),
            "jitter": float(self.jitter),
        }


_KNOWN_DATASET_RUNTIMES: dict[str, dict[str, Any]] = {
    "intent_dataset.csv": {
        "seconds": 7600.0,
        "epochs": 1.0,
        "confidence": 1.0,
        "description": "User-reported full dataset epoch duration",
    }
}


def _normalise_dataset_key(value: str | None) -> str | None:
    """Normalise ``value`` (path or label) into a lookup key."""

    if not value:
        return None
    candidate = value.strip().lower()
    if not candidate:
        return None
    # Treat values that look like file paths by extracting their basename.
    if "/" in candidate or candidate.endswith(".csv"):
        candidate = Path(candidate).name.lower()
    return candidate


def _resolve_known_dataset_runtime(
    dataset_summary: DatasetSummary | None,
    *,
    label: str | None = None,
) -> dict[str, Any] | None:
    """Return a known runtime anchor for recognised datasets, if available."""

    candidates: list[str] = []
    if dataset_summary is not None and dataset_summary.path:
        candidates.append(dataset_summary.path)
    if label:
        candidates.append(label)
    for candidate in candidates:
        key = _normalise_dataset_key(candidate)
        if key is None:
            continue
        anchor = _KNOWN_DATASET_RUNTIMES.get(key)
        if anchor:
            resolved = dict(anchor)
            resolved["key"] = key
            resolved["seconds"] = float(resolved.get("seconds", 0.0) or 0.0)
            resolved["epochs"] = float(resolved.get("epochs", 1.0) or 1.0)
            resolved["confidence"] = float(resolved.get("confidence", 1.0) or 0.0)
            return resolved
    return None


@dataclass
class SpeedTestCalibration:
    """User-provided calibration to anchor runtime projections."""

    examples: int
    seconds: float
    epochs: float = 1.0

    def compute_scale(
        self,
        passes_per_example: float | None,
        per_pass_seconds: float | None,
    ) -> float | None:
        if self.examples <= 0 or self.seconds <= 0:
            return None
        if passes_per_example is None or passes_per_example <= 0:
            return None
        if per_pass_seconds is None or per_pass_seconds <= 0:
            return None
        total_passes = passes_per_example * float(self.examples) * max(self.epochs, 0.0)
        if total_passes <= 0:
            return None
        estimated = per_pass_seconds * total_passes
        if estimated <= 0:
            return None
        return float(self.seconds) / estimated


def _summarise_token_stats(
    texts: Sequence[str],
    *,
    sample_size: int = 2048,
    tokeniser: Callable[[str], Sequence[str]] | None = None,
) -> tuple[float, int, float]:
    """Summarise token statistics for ``texts`` using an optional ``tokeniser``."""

    if not texts:
        return 0.0, 0, 0.0
    sample_size = max(1, min(len(texts), sample_size))
    step = max(1, len(texts) // sample_size)
    total_tokens = 0.0
    sampled = 0
    for index in range(0, len(texts), step):
        if sampled >= sample_size:
            break
        text = texts[index]
        if tokeniser is None:
            token_count = len(text.split())
        else:
            token_count = len(tokeniser(text))
        total_tokens += float(token_count)
        sampled += 1
    if sampled == 0:
        return 0.0, 0, 0.0
    average = total_tokens / float(sampled)
    total_estimate = average * float(len(texts))
    return average, sampled, total_estimate


def _build_dataset_summary_from_texts(
    texts: Sequence[str],
    *,
    path: Path | None = None,
    sample_limit: int | None = None,
    tokeniser: Callable[[str], Sequence[str]] | None = None,
) -> DatasetSummary:
    """Construct a :class:`DatasetSummary` from in-memory ``texts``."""

    summary = DatasetSummary(path=str(path) if path is not None else None)
    summary.examples = len(texts)
    if not texts:
        return summary
    if sample_limit is None or sample_limit <= 0:
        sample_size = min(len(texts), 4096)
    else:
        sample_size = min(len(texts), sample_limit)
    sample_size = max(1, sample_size)
    average, sampled, total_estimate = _summarise_token_stats(
        texts,
        sample_size=sample_size,
        tokeniser=tokeniser,
    )
    summary.sample_size = sample_size
    summary.token_samples = sampled
    summary.average_tokens = average
    summary.total_tokens = total_estimate
    summary.truncated = sampled < summary.examples
    return summary

def _estimate_training_flops(args: argparse.Namespace, average_tokens: float) -> float:
    """Heuristically estimate forward/backward FLOPs per example pass."""

    tokens = max(1.0, average_tokens)
    encoder_type = getattr(args, "encoder_type", "transformer")
    if encoder_type == "bilstm":
        hidden = max(1.0, float(getattr(args, "hidden_dim", 256)))
        layers = max(1.0, float(getattr(args, "encoder_layers", 2)))
        directions = 2.0
        gate_cost = 8.0  # input, forget, cell, output gates
        base = gate_cost * directions * layers * tokens * hidden * hidden
        if getattr(args, "bilstm_conv_head", False):
            conv_channels = max(1.0, float(getattr(args, "bilstm_conv_channels", hidden)))
            kernel_list = getattr(args, "bilstm_conv_kernels", [3, 5, 7]) or [3, 5, 7]
            kernel_flops = sum(max(1.0, float(kernel)) for kernel in kernel_list)
            base += conv_channels * tokens * kernel_flops
        return base * 3.0  # roughly account for backward + optimizer
    if encoder_type == "st":
        hidden = max(1.0, float(getattr(args, "st_hidden_dim", 768)))
        layers = max(1.0, float(getattr(args, "st_mlp_layers", 4)))
        expansion = max(1.0, float(getattr(args, "st_mlp_expansion", 1.6)))
        seq = max(tokens, 8.0)
        ffn_hidden = hidden * expansion
        layer_cost = 2.0 * seq * hidden * ffn_hidden * 2.0
        moe_experts = max(0, int(getattr(args, "st_moe_experts", 0)))
        if moe_experts >= 2:
            moe_hidden = max(hidden, float(getattr(args, "st_moe_hidden_dim", hidden)))
            layer_cost += seq * hidden * moe_hidden * moe_experts
        return layer_cost * layers * 3.0
    # transformer-like default
    hidden = max(1.0, float(getattr(args, "hidden_dim", getattr(args, "embedding_dim", 768))))
    layers = max(1.0, float(getattr(args, "encoder_layers", 12)))
    heads = max(1.0, float(getattr(args, "attention_heads", max(1, int(hidden // 64)))))
    ffn_dim = max(hidden, float(getattr(args, "ffn_dim", hidden * 4)))
    seq = min(max(tokens * 1.2, 8.0), float(getattr(args, "max_seq_len", max(32, int(tokens * 1.5)))))
    d_head = hidden / heads
    qkv = 3.0 * seq * hidden * hidden
    attn_scores = heads * seq * seq * d_head
    attn_output = seq * hidden * hidden
    ffn_cost = 2.0 * seq * hidden * ffn_dim
    layer_flops = qkv + attn_scores + attn_output + ffn_cost
    return layer_flops * layers * 3.2  # forward + backward + optimiser


def _resolve_reference_gflops(
    args: argparse.Namespace, summary: Mapping[str, Any] | None = None
) -> float | None:
    """Resolve the sustained GFLOP/s estimate from CLI flags or simulation output."""

    manual = float(getattr(args, "speed_test_reference_gflops", 0.0) or 0.0)
    if manual > 0:
        return manual
    if summary and summary.get("enabled"):
        for key in ("mean_gflops", "max_gflops"):
            value = summary.get(key)
            if value is not None:
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    return parsed
    return None


def _build_speed_test_profile(
    args: argparse.Namespace,
    *,
    average_tokens: float,
    reference_gflops: float | None,
    fallback_mode: bool,
    observed_per_pass: float | None = None,
    observed_tokens_per_pass: float | None = None,
) -> SpeedTestComplexityProfile:
    flops_per_pass = _estimate_training_flops(args, average_tokens)
    notes: dict[str, Any] = {
        "avg_tokens": float(average_tokens),
        "estimated_flops_per_pass": float(flops_per_pass),
    }
    resolved_gflops = reference_gflops if reference_gflops and reference_gflops > 0 else None
    if resolved_gflops is None:
        resolved_gflops = 275.0  # default sustained throughput for projection
        notes["reference_gflops_source"] = "default"
    notes["reference_gflops"] = float(resolved_gflops)
    override = None
    if flops_per_pass > 0 and resolved_gflops:
        override = flops_per_pass / (resolved_gflops * 1e9)
        notes["override_pass_seconds_raw"] = override
    baseline_multiplier = 1.0
    if fallback_mode and override is None:
        baseline_multiplier = min(1024.0, max(1.0, flops_per_pass / max(average_tokens * 2048.0, 1.0)))
        notes["fallback_multiplier"] = baseline_multiplier
    if observed_per_pass is not None and observed_per_pass > 0:
        notes["observed_per_pass_seconds"] = float(observed_per_pass)
        if override is not None and override > 0:
            calibration = observed_per_pass / override
            notes["override_calibration"] = float(calibration)
            override *= calibration
    if observed_tokens_per_pass is not None and observed_tokens_per_pass > 0:
        notes["observed_tokens_per_pass"] = float(observed_tokens_per_pass)
    if override is not None and override > 0:
        notes["override_pass_seconds"] = float(override)
    return SpeedTestComplexityProfile(
        baseline_multiplier=baseline_multiplier,
        override_pass_seconds=override,
        constant_overhead_seconds=0.0,
        notes=notes,
    )


def _build_speed_test_calibration(args: argparse.Namespace) -> SpeedTestCalibration | None:
    seconds = float(getattr(args, "speed_test_calibration_seconds", 0.0) or 0.0)
    examples = int(getattr(args, "speed_test_calibration_examples", 0) or 0)
    epochs = float(getattr(args, "speed_test_calibration_epochs", 1.0) or 1.0)
    if seconds > 0 and examples > 0 and epochs > 0:
        return SpeedTestCalibration(examples=examples, seconds=seconds, epochs=epochs)
    return None


class SpeedTestLogger:
    """Collect wall-clock timings and throughput statistics for speed tests."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._sections: dict[str, dict[str, Any]] = {}
        self._folds: list[dict[str, Any]] = []
        self._total_passes: float = 0.0
        self._total_examples: int = 0
        self._total_elapsed: float | None = None
        self._start = time.perf_counter() if self.enabled else 0.0
        self._estimates: list[dict[str, Any]] = []
        self._complexity_profile: SpeedTestComplexityProfile | None = None
        self._calibration: SpeedTestCalibration | None = None
        self._complexity_notes: dict[str, Any] = {}
        self._training_elapsed: float = 0.0
        self._total_tokens: float = 0.0
        self._epoch_details: list[dict[str, Any]] = []
        self._using_epoch_details: bool = False
        self._observed_epoch_examples: float = 0.0

    def marker(self) -> float:
        """Return a perf_counter marker used to delimit timing sections."""

        return time.perf_counter()

    def record_section(
        self,
        name: str,
        start: float,
        *,
        count: float | None = None,
        passes: float | None = None,
        notes: Mapping[str, float] | None = None,
        add_to_total: bool = True,
    ) -> None:
        if not self.enabled:
            return
        elapsed = max(0.0, time.perf_counter() - start)
        entry: dict[str, Any] = {"seconds": elapsed}
        if count is not None:
            entry["count"] = float(count)
            if elapsed > 0:
                entry["per_second"] = float(count) / elapsed
        if passes is not None:
            passes_value = float(passes)
            entry["example_passes"] = passes_value
            if elapsed > 0 and passes_value > 0:
                entry["passes_per_second"] = passes_value / elapsed
            if add_to_total:
                self._total_passes += passes_value
        if notes:
            for key, value in notes.items():
                entry[key] = float(value)
        self._sections[name] = entry

    def record_fold(
        self,
        fold_index: int,
        start: float,
        *,
        examples: int | None = None,
        epochs: float | None = None,
        passes: float | None = None,
        add_to_total: bool = True,
    ) -> None:
        if not self.enabled:
            return
        elapsed = max(0.0, time.perf_counter() - start)
        entry: dict[str, Any] = {"fold": float(fold_index), "seconds": elapsed}
        if examples is not None:
            entry["examples"] = float(examples)
        if epochs is not None:
            entry["epochs"] = float(epochs)
        passes_value: float | None
        if passes is not None:
            passes_value = float(passes)
        elif examples is not None and epochs is not None:
            passes_value = float(examples) * float(epochs)
        else:
            passes_value = None
        if passes_value is not None:
            entry["example_passes"] = passes_value
            if elapsed > 0 and passes_value > 0:
                entry["passes_per_second"] = passes_value / elapsed
            if add_to_total:
                self._total_passes += passes_value
        self._folds.append(entry)

    def finish(self, *, total_examples: int, extra_passes: float = 0.0) -> None:
        if not self.enabled:
            return
        self._total_examples = int(total_examples)
        if extra_passes:
            self._total_passes += float(extra_passes)
        elapsed_total = max(0.0, time.perf_counter() - self._start)
        self._total_elapsed = elapsed_total
        if self._training_elapsed <= 0:
            dataset_section = self._sections.get("dataset_load")
            dataset_seconds = dataset_section.get("seconds") if dataset_section else None
            candidate = elapsed_total
            if dataset_seconds is not None:
                candidate = max(candidate - dataset_seconds, 0.0)
            self._training_elapsed = max(self._training_elapsed, candidate)

    def passes_per_example(self) -> float | None:
        if not self.enabled:
            return None
        if self._total_examples <= 0:
            return None
        if self._total_passes <= 0:
            return None
        return self._total_passes / float(self._total_examples)

    def baseline_per_pass(self) -> float | None:
        if not self.enabled:
            return None
        if self._total_passes <= 0:
            return None
        if self._training_elapsed > 0:
            return self._training_elapsed / self._total_passes
        if self._total_elapsed is None:
            return None
        training_seconds = self._total_elapsed
        dataset_section = self._sections.get("dataset_load")
        dataset_seconds = dataset_section.get("seconds") if dataset_section else None
        if dataset_seconds is not None:
            training_seconds = max(training_seconds - dataset_seconds, 0.0)
        if training_seconds <= 0:
            return None
        return training_seconds / self._total_passes

    def per_token_seconds(self) -> float | None:
        if not self.enabled:
            return None
        if self._total_tokens <= 0:
            return None
        if self._training_elapsed <= 0:
            return None
        return self._training_elapsed / self._total_tokens

    def average_tokens_per_pass(self) -> float | None:
        if not self.enabled:
            return None
        if self._total_tokens <= 0:
            return None
        if self._total_passes <= 0:
            return None
        return self._total_tokens / self._total_passes

    def using_epoch_details(self) -> bool:
        return self._using_epoch_details

    def apply_complexity_profile(self, profile: SpeedTestComplexityProfile) -> None:
        if not self.enabled:
            return
        self._complexity_profile = profile
        self._complexity_notes = dict(profile.notes)

    def configure_calibration(self, calibration: SpeedTestCalibration) -> None:
        if not self.enabled:
            return
        self._calibration = calibration

    def record_training_epoch(
        self,
        *,
        stage: str,
        epoch: int,
        seconds: float,
        examples: float,
        tokens: float,
        batches: float,
        passes: float | None = None,
        add_to_total: bool = True,
    ) -> None:
        if not self.enabled:
            return
        seconds = max(0.0, float(seconds))
        examples = max(0.0, float(examples))
        tokens = max(0.0, float(tokens))
        batches = max(0.0, float(batches))
        if passes is None:
            passes_value = examples
        else:
            passes_value = max(0.0, float(passes))
        entry = {
            "stage": stage,
            "epoch": float(epoch),
            "seconds": seconds,
            "examples": examples,
            "tokens": tokens,
            "batches": batches,
            "passes": passes_value,
        }
        self._epoch_details.append(entry)
        self._using_epoch_details = True
        if add_to_total and passes_value > 0:
            self._total_passes += passes_value
        if tokens > 0:
            self._total_tokens += tokens
        if seconds > 0:
            self._training_elapsed += seconds
        if examples > 0:
            self._observed_epoch_examples += examples

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds <= 0:
            return "0m 0s"
        total = max(0.0, seconds)
        hours = int(total // 3600)
        minutes = int((total % 3600) // 60)
        remaining = total - hours * 3600 - minutes * 60
        if hours > 0:
            return f"{hours}h {minutes}m {remaining:.1f}s"
        return f"{minutes}m {remaining:.1f}s"

    @staticmethod
    def _blend_component_values(
        components: Sequence[SpeedTestEstimateComponent],
    ) -> tuple[float | None, dict[str, Any]]:
        filtered = [
            component
            for component in components
            if component.seconds > 0 and component.adjusted_weight > 0
        ]
        if not filtered:
            return None, {}
        total_weight = sum(component.adjusted_weight for component in filtered)
        if total_weight <= 0:
            return None, {}

        arithmetic = sum(
            component.seconds * component.adjusted_weight for component in filtered
        ) / total_weight
        harmonic_denominator = sum(
            component.adjusted_weight / component.seconds
            for component in filtered
            if component.seconds > 0
        )
        harmonic = (
            total_weight / harmonic_denominator if harmonic_denominator > 0 else arithmetic
        )
        geometric = math.exp(
            sum(component.adjusted_weight * math.log(component.seconds) for component in filtered)
            / total_weight
        )
        quadratic = math.sqrt(
            sum(
                component.adjusted_weight * (component.seconds ** 2)
                for component in filtered
            )
            / total_weight
        )
        sorted_components = sorted(filtered, key=lambda item: item.seconds)
        cumulative = 0.0
        median_value = sorted_components[-1].seconds
        for component in sorted_components:
            cumulative += component.adjusted_weight
            if cumulative >= total_weight * 0.5:
                median_value = component.seconds
                break
        lower_cut = total_weight * 0.1
        upper_cut = total_weight * 0.9
        trimmed_sum = 0.0
        trimmed_weight = 0.0
        cumulative = 0.0
        for component in sorted_components:
            start = cumulative
            end = cumulative + component.adjusted_weight
            overlap = max(0.0, min(end, upper_cut) - max(start, lower_cut))
            if overlap > 0:
                trimmed_sum += component.seconds * overlap
                trimmed_weight += overlap
            cumulative = end
        trimmed_mean = trimmed_sum / trimmed_weight if trimmed_weight > 0 else arithmetic
        extrema_min = sorted_components[0].seconds
        extrema_max = sorted_components[-1].seconds
        variance = sum(
            component.adjusted_weight * (component.seconds - arithmetic) ** 2
            for component in filtered
        ) / total_weight
        std_dev = math.sqrt(max(variance, 0.0))
        coefficient_variation = (
            std_dev / arithmetic if arithmetic > 0 else 0.0
        )
        stability = 1.0 / (1.0 + coefficient_variation * coefficient_variation)
        confidence_mean = sum(
            component.confidence * component.adjusted_weight for component in filtered
        ) / total_weight
        evidence_total = sum(component.evidence for component in filtered)
        probabilities = [
            component.adjusted_weight / total_weight for component in filtered if component.adjusted_weight > 0
        ]
        entropy = -sum(prob * math.log(prob) for prob in probabilities if prob > 0)
        max_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 1.0
        jitter = sum(
            component.jitter * component.adjusted_weight for component in filtered
        ) / total_weight

        blended = (
            arithmetic * 0.22
            + geometric * 0.18
            + harmonic * 0.12
            + quadratic * 0.08
            + median_value * 0.22
            + trimmed_mean * 0.18
        )
        extremal_blend = (extrema_min + extrema_max) * 0.5
        blended = blended * 0.9 + extremal_blend * 0.1
        blended *= 0.75 + 0.25 * stability
        blended *= 0.85 + 0.15 * min(1.0, confidence_mean)
        evidence_term = 1.0 - math.exp(-evidence_total / max(len(filtered), 1))
        blended *= 0.88 + 0.12 * min(1.0, evidence_term)
        blended *= 0.9 + 0.1 * entropy_ratio
        blended *= 1.0 + jitter

        anchor_component = None
        for component in filtered:
            if component.source == "dataset_anchor":
                anchor_component = component
                break
        anchor_pull = 0.0
        if anchor_component is not None:
            anchor_pull = min(
                0.65,
                anchor_component.confidence * (0.55 + 0.15 * stability),
            )
            blended = (
                blended * (1.0 - anchor_pull)
                + anchor_component.seconds * anchor_pull
            )

        notes = {
            "blend_arithmetic": arithmetic,
            "blend_geometric": geometric,
            "blend_harmonic": harmonic,
            "blend_quadratic": quadratic,
            "blend_median": median_value,
            "blend_trimmed_mean": trimmed_mean,
            "blend_min": extrema_min,
            "blend_max": extrema_max,
            "blend_variance": variance,
            "blend_std": std_dev,
            "blend_cv": coefficient_variation,
            "blend_stability": stability,
            "blend_confidence": confidence_mean,
            "blend_evidence": evidence_total,
            "blend_entropy": entropy,
            "blend_entropy_ratio": entropy_ratio,
            "blend_weight_sum": total_weight,
            "blend_component_count": float(len(filtered)),
            "blend_anchor_pull": anchor_pull,
        }
        return blended, notes

    def register_estimate(
        self,
        label: str,
        target_examples: int,
        *,
        observed_examples: int | None = None,
        target_average_tokens: float | None = None,
        target_total_tokens: float | None = None,
        observed_average_tokens: float | None = None,
        observed_total_tokens: float | None = None,
        dataset_summary: DatasetSummary | None = None,
        observed_dataset_summary: DatasetSummary | None = None,
    ) -> None:
        if not self.enabled:
            return
        if self._total_elapsed is None:
            return
        if target_examples <= 0:
            return
        observed_pool = (
            float(self._observed_epoch_examples)
            if self._observed_epoch_examples > 0
            else float(self._total_examples)
        )
        observed = (
            float(observed_examples)
            if observed_examples is not None and observed_examples > 0
            else observed_pool
        )
        if observed <= 0:
            return

        multiplier = float(target_examples) / float(observed)
        dataset_section = self._sections.get("dataset_load")
        dataset_seconds = dataset_section.get("seconds") if dataset_section else None
        dataset_count = dataset_section.get("count") if dataset_section else None
        if dataset_seconds is not None and dataset_count and dataset_count > 0:
            dataset_rate = dataset_seconds / float(dataset_count)
            estimated_dataset_seconds = dataset_rate * float(target_examples)
        else:
            dataset_seconds = None
            estimated_dataset_seconds = None

        training_seconds = self._total_elapsed
        if dataset_seconds is not None:
            training_seconds = max(self._total_elapsed - dataset_seconds, 0.0)

        baseline_per_pass = self.baseline_per_pass()
        per_pass_seconds = baseline_per_pass
        pass_seconds_source = "baseline"
        if self._complexity_profile is not None:
            resolved = self._complexity_profile.resolve_pass_seconds(baseline_per_pass)
            if resolved is not None:
                per_pass_seconds = resolved
                pass_seconds_source = "complexity"
        calibration_scale = None
        passes_per_example = self.passes_per_example()
        if self._calibration is not None:
            calibration_scale = self._calibration.compute_scale(passes_per_example, per_pass_seconds)
            if calibration_scale is not None and calibration_scale > 0:
                per_pass_seconds = per_pass_seconds * calibration_scale if per_pass_seconds is not None else None
        per_token_seconds = self.per_token_seconds()
        tokens_per_pass = self.average_tokens_per_pass()
        target_tokens_total = (
            float(target_total_tokens)
            if target_total_tokens is not None and target_total_tokens > 0
            else None
        )
        observed_tokens_total = (
            float(observed_total_tokens)
            if observed_total_tokens is not None and observed_total_tokens > 0
            else (self._total_tokens if self._total_tokens > 0 else None)
        )
        observed_avg_tokens = (
            float(observed_average_tokens)
            if observed_average_tokens is not None and observed_average_tokens > 0
            else (
                (observed_tokens_total / observed)
                if observed_tokens_total is not None and observed > 0
                else None
            )
        )
        if observed_avg_tokens is None and observed_dataset_summary is not None:
            if observed_dataset_summary.average_tokens > 0:
                observed_avg_tokens = float(observed_dataset_summary.average_tokens)
        if observed_tokens_total is None and observed_dataset_summary is not None:
            if observed_dataset_summary.total_tokens > 0:
                observed_tokens_total = float(observed_dataset_summary.total_tokens)
        target_avg_tokens = (
            float(target_average_tokens)
            if target_average_tokens is not None and target_average_tokens > 0
            else None
        )
        token_scale = None
        if (
            target_avg_tokens is not None
            and observed_avg_tokens is not None
            and observed_avg_tokens > 0
        ):
            token_scale = target_avg_tokens / observed_avg_tokens
        tokens_per_pass_adjusted: float | None = None
        if tokens_per_pass is not None and tokens_per_pass > 0:
            tokens_per_pass_adjusted = tokens_per_pass
            if token_scale is not None and token_scale > 0:
                tokens_per_pass_adjusted *= token_scale
        target_passes = self._total_passes * multiplier if self._total_passes > 0 else None
        if (
            per_token_seconds is not None
            and per_token_seconds > 0
            and tokens_per_pass_adjusted is not None
            and tokens_per_pass_adjusted > 0
        ):
            per_pass_seconds = per_token_seconds * tokens_per_pass_adjusted
            pass_seconds_source = "tokens_scaled" if token_scale else "tokens"
        elif (
            per_token_seconds is not None
            and per_token_seconds > 0
            and target_tokens_total is not None
            and target_passes is not None
            and target_passes > 0
        ):
            per_pass_seconds = (per_token_seconds * target_tokens_total) / target_passes
            pass_seconds_source = "tokens_total"
        per_pass_components: list[SpeedTestEstimateComponent] = []
        per_pass_candidates: list[dict[str, Any]] = []

        def _add_component(
            value: float | None,
            weight: float,
            source: str,
            *,
            confidence: float = 1.0,
            evidence: float = 1.0,
            jitter: float = 0.0,
        ) -> None:
            if value is None or value <= 0:
                return
            base_weight = max(0.0, float(weight))
            if base_weight <= 0:
                return
            confidence_clamped = max(0.0, min(float(confidence), 1.0))
            evidence_value = max(0.0, float(evidence))
            adjusted_weight = base_weight * (0.5 + 0.5 * confidence_clamped) * (1.0 + math.log1p(evidence_value))
            component = SpeedTestEstimateComponent(
                source=source,
                seconds=float(value),
                weight=base_weight,
                confidence=confidence_clamped,
                evidence=evidence_value,
                adjusted_weight=adjusted_weight,
                jitter=float(jitter),
            )
            per_pass_components.append(component)
            per_pass_candidates.append(component.to_dict())

        pass_seconds_source = "ensemble_advanced"
        calibration_scale = None
        baseline_candidate = None
        if baseline_per_pass is not None and baseline_per_pass > 0:
            baseline_candidate = baseline_per_pass
            if self._calibration is not None:
                calibration_scale = self._calibration.compute_scale(passes_per_example, baseline_per_pass)
                if calibration_scale is not None and calibration_scale > 0:
                    baseline_candidate = baseline_per_pass * calibration_scale
            baseline_confidence = 0.35 + 0.65 * min(1.0, float(observed) / float(target_examples))
            baseline_evidence = max(1.0, float(self._total_passes) if self._total_passes > 0 else float(observed))
            _add_component(
                baseline_candidate,
                1.2,
                "baseline",
                confidence=baseline_confidence,
                evidence=baseline_evidence,
            )

        complexity_candidate = None
        if self._complexity_profile is not None:
            resolved = self._complexity_profile.resolve_pass_seconds(baseline_per_pass)
            if resolved is not None and resolved > 0:
                complexity_candidate = float(resolved)
                _add_component(
                    complexity_candidate,
                    1.5,
                    "complexity",
                    confidence=0.75,
                    evidence=max(1.0, float(target_examples)),
                )

        tokens_candidate = None
        if (
            per_token_seconds is not None
            and per_token_seconds > 0
            and tokens_per_pass_adjusted is not None
            and tokens_per_pass_adjusted > 0
        ):
            tokens_candidate = per_token_seconds * tokens_per_pass_adjusted
        elif (
            per_token_seconds is not None
            and per_token_seconds > 0
            and target_tokens_total is not None
            and target_passes is not None
            and target_passes > 0
        ):
            tokens_candidate = (per_token_seconds * target_tokens_total) / target_passes
        if tokens_candidate is not None and tokens_candidate > 0:
            token_evidence = 0.0
            if observed_dataset_summary is not None and observed_dataset_summary.token_samples > 0:
                token_evidence = float(observed_dataset_summary.token_samples)
            elif observed_tokens_total is not None and observed_tokens_total > 0 and observed_avg_tokens:
                token_evidence = observed_tokens_total / max(observed_avg_tokens, 1e-9)
            token_confidence = 0.3
            if token_scale is not None and token_scale > 0:
                token_confidence += min(0.4, token_scale * 0.2)
            if observed_avg_tokens is not None and observed_avg_tokens > 0:
                token_confidence += 0.15
            if observed_tokens_total is not None and observed_tokens_total > 0:
                token_confidence += 0.1
            token_confidence = min(token_confidence, 0.95)
            _add_component(
                tokens_candidate,
                1.8,
                "tokens_scaled" if token_scale else "tokens",
                confidence=token_confidence,
                evidence=max(1.0, token_evidence),
            )

        dataset_anchor = _resolve_known_dataset_runtime(dataset_summary, label=label)
        anchor_seconds = None
        anchor_epochs = None
        anchor_passes = None
        anchor_per_pass = None
        if dataset_anchor is not None:
            anchor_seconds = float(dataset_anchor.get("seconds", 0.0) or 0.0)
            anchor_epochs = float(dataset_anchor.get("epochs", 1.0) or 1.0)
            anchor_confidence = float(dataset_anchor.get("confidence", 1.0) or 0.0)
            if anchor_seconds > 0 and target_examples > 0:
                passes_reference = passes_per_example
                if (passes_reference is None or passes_reference <= 0) and target_passes is not None and target_passes > 0:
                    passes_reference = target_passes / float(target_examples)
                if passes_reference is not None and passes_reference > 0:
                    anchor_passes = passes_reference * float(target_examples) * max(anchor_epochs, 1e-9)
                    if anchor_passes > 0:
                        anchor_per_pass = anchor_seconds / anchor_passes
                        anchor_weight = 6.0 + math.log10(max(float(target_examples), 10.0))
                        _add_component(
                            anchor_per_pass,
                            anchor_weight,
                            "dataset_anchor",
                            confidence=min(1.0, anchor_confidence),
                            evidence=max(1.0, anchor_passes),
                            jitter=0.02,
                        )

        per_pass_seconds, blend_notes = self._blend_component_values(per_pass_components)
        if per_pass_seconds is None:
            pass_seconds_source = "fallback"
            if baseline_candidate is not None:
                per_pass_seconds = baseline_candidate
            elif tokens_candidate is not None:
                per_pass_seconds = tokens_candidate
            elif anchor_per_pass is not None:
                per_pass_seconds = anchor_per_pass
            else:
                per_pass_seconds = baseline_per_pass

        estimated_training_seconds_raw: float | None = None
        if (
            per_pass_seconds is not None
            and per_pass_seconds > 0
            and target_passes is not None
            and target_passes > 0
        ):
            estimated_training_seconds_raw = per_pass_seconds * target_passes

        tokens_training_candidate = None
        if (
            target_tokens_total is not None
            and target_tokens_total > 0
            and per_token_seconds is not None
            and per_token_seconds > 0
        ):
            tokens_training_candidate = per_token_seconds * target_tokens_total
        scaled_elapsed_candidate = None
        if training_seconds > 0 and multiplier > 0:
            scaled_elapsed_candidate = training_seconds * multiplier

        dataset_anchor_seconds = None
        dataset_anchor_scale = None
        dataset_anchor_confidence = float(dataset_anchor.get("confidence", 0.0)) if dataset_anchor is not None else 0.0
        if dataset_anchor is not None and anchor_seconds is not None and anchor_seconds > 0:
            if anchor_passes is not None and anchor_passes > 0 and target_passes is not None and target_passes > 0:
                dataset_anchor_seconds = anchor_seconds * (target_passes / anchor_passes)
            else:
                dataset_anchor_seconds = anchor_seconds * max(multiplier, 1.0)
            if dataset_anchor_seconds < anchor_seconds:
                dataset_anchor_seconds = anchor_seconds

        training_components: list[SpeedTestEstimateComponent] = []
        training_candidate_dicts: list[dict[str, Any]] = []

        def _add_training_candidate(
            value: float | None,
            weight: float,
            source: str,
            *,
            confidence: float = 1.0,
            evidence: float = 1.0,
        ) -> None:
            if value is None or value <= 0:
                return
            base_weight = max(0.0, float(weight))
            if base_weight <= 0:
                return
            confidence_clamped = max(0.0, min(float(confidence), 1.0))
            evidence_value = max(0.0, float(evidence))
            adjusted_weight = base_weight * (0.5 + 0.5 * confidence_clamped) * (1.0 + math.log1p(evidence_value))
            component = SpeedTestEstimateComponent(
                source=source,
                seconds=float(value),
                weight=base_weight,
                confidence=confidence_clamped,
                evidence=evidence_value,
                adjusted_weight=adjusted_weight,
            )
            training_components.append(component)
            training_candidate_dicts.append(component.to_dict())

        if estimated_training_seconds_raw is not None:
            per_pass_weight = blend_notes.get("blend_weight_sum", float(len(per_pass_components))) if blend_notes else float(len(per_pass_components))
            per_pass_confidence = blend_notes.get("blend_confidence", 0.0) if blend_notes else 0.5
            per_pass_evidence = blend_notes.get("blend_evidence", 0.0) if blend_notes else 0.0
            _add_training_candidate(
                estimated_training_seconds_raw,
                max(1.0, per_pass_weight),
                pass_seconds_source,
                confidence=max(0.3, min(1.0, per_pass_confidence if per_pass_confidence > 0 else 0.6)),
                evidence=max(1.0, per_pass_evidence),
            )
        if tokens_training_candidate is not None:
            token_confidence_training = 0.4
            if token_scale is not None and token_scale > 0:
                token_confidence_training += min(0.4, token_scale * 0.25)
            if observed_tokens_total is not None and observed_tokens_total > 0:
                token_confidence_training += 0.1
            _add_training_candidate(
                tokens_training_candidate,
                1.6,
                "tokens_direct",
                confidence=min(0.95, token_confidence_training),
                evidence=max(1.0, float(target_tokens_total) if target_tokens_total else 1.0),
            )
        if scaled_elapsed_candidate is not None:
            elapsed_confidence = min(0.95, 0.4 + 0.6 * min(1.0, float(observed) / float(target_examples)))
            _add_training_candidate(
                scaled_elapsed_candidate,
                1.1,
                "scaled_elapsed",
                confidence=elapsed_confidence,
                evidence=max(1.0, training_seconds),
            )
        if dataset_anchor_seconds is not None:
            _add_training_candidate(
                dataset_anchor_seconds,
                4.8,
                "dataset_anchor",
                confidence=min(1.0, dataset_anchor_confidence if dataset_anchor_confidence > 0 else 0.85),
                evidence=max(1.0, anchor_passes or float(target_examples)),
            )

        estimated_training_seconds, training_blend_notes = self._blend_component_values(training_components)
        if estimated_training_seconds is None:
            if dataset_anchor_seconds is not None:
                estimated_training_seconds = dataset_anchor_seconds
                pass_seconds_source = "dataset_anchor_fallback"
            elif estimated_training_seconds_raw is not None:
                estimated_training_seconds = estimated_training_seconds_raw
            elif tokens_training_candidate is not None:
                estimated_training_seconds = tokens_training_candidate
                pass_seconds_source = "tokens_direct"
            else:
                estimated_training_seconds = scaled_elapsed_candidate
                if scaled_elapsed_candidate is not None:
                    pass_seconds_source = "scaled_elapsed"

        if (
            estimated_training_seconds is None
            and scaled_elapsed_candidate is None
            and estimated_training_seconds_raw is None
            and dataset_anchor_seconds is not None
        ):
            estimated_training_seconds = dataset_anchor_seconds

        if (
            dataset_anchor_seconds is not None
            and dataset_anchor_seconds > 0
            and estimated_training_seconds is not None
            and estimated_training_seconds > 0
        ):
            dataset_anchor_scale = dataset_anchor_seconds / estimated_training_seconds

        estimated_total = 0.0
        if estimated_dataset_seconds is not None:
            estimated_total += estimated_dataset_seconds
        if estimated_training_seconds is not None:
            estimated_total += estimated_training_seconds
        else:
            estimated_total += self._total_elapsed * multiplier

        self._estimates.append(
            {
                "label": label,
                "target_examples": float(target_examples),
                "observed_examples": float(observed),
                "multiplier": multiplier,
                "estimated_seconds": estimated_total,
                "estimated_dataset_seconds": estimated_dataset_seconds or 0.0,
                "estimated_training_seconds": estimated_training_seconds or 0.0,
                "per_pass_seconds": per_pass_seconds or 0.0,
                "pass_seconds_source": pass_seconds_source,
                "calibration_scale": calibration_scale or 0.0,
                "per_token_seconds": per_token_seconds or 0.0,
                "tokens_per_pass": tokens_per_pass or 0.0,
                "tokens_per_pass_adjusted": tokens_per_pass_adjusted or 0.0,
                "target_average_tokens": target_avg_tokens or 0.0,
                "target_total_tokens": target_tokens_total or 0.0,
                "observed_average_tokens": observed_avg_tokens or 0.0,
                "observed_total_tokens": observed_tokens_total or 0.0,
                "token_scale": token_scale or 0.0,
                "per_pass_candidates": per_pass_candidates,
                "blend_notes": blend_notes,
                "training_candidates": training_candidate_dicts,
                "training_blend_notes": training_blend_notes,
                "dataset_anchor_seconds": dataset_anchor_seconds or 0.0,
                "dataset_anchor_epochs": anchor_epochs or 0.0 if anchor_epochs is not None else 0.0,
                "dataset_anchor_scale": dataset_anchor_scale or 0.0,
                "dataset_anchor_per_pass": anchor_per_pass or 0.0,
                "dataset_anchor_confidence": dataset_anchor_confidence,
                "dataset_anchor_key": dataset_anchor.get("key") if dataset_anchor is not None else "",
            }
        )

    def report(self) -> None:
        if not self.enabled:
            return
        print("\nSpeed test summary:")
        dataset = self._sections.get("dataset_load")
        if dataset:
            count = int(dataset.get("count", 0))
            seconds = dataset.get("seconds", 0.0)
            rate = dataset.get("per_second")
            if rate:
                print(
                    f"- Dataset loading: {seconds:.2f}s for {count} examples "
                    f"({rate:,.1f} samples/s)"
                )
            else:
                print(f"- Dataset loading: {seconds:.2f}s for {count} examples")
            avg_tokens = dataset.get("average_tokens")
            token_samples = dataset.get("token_samples")
            estimated_tokens = dataset.get("estimated_tokens")
            if avg_tokens and token_samples is not None:
                sampled_count = int(token_samples)
                token_line = (
                    f"  · Token profile: {float(avg_tokens):,.1f} avg tokens/example"
                )
                if sampled_count:
                    if sampled_count < count:
                        token_line += f" (sampled {sampled_count} of {count})"
                    else:
                        token_line += f" (sampled {sampled_count})"
                if estimated_tokens:
                    token_line += f"; ~{float(estimated_tokens):,.0f} total tokens"
                print(token_line)

        cross = self._sections.get("cross_validation")
        if cross:
            total_seconds = cross.get("seconds", 0.0)
            fold_count = int(cross.get("count", len(self._folds)))
            plural = "s" if fold_count != 1 else ""
            print(
                f"- Cross-validation: {total_seconds:.2f}s across {fold_count} fold{plural}"
            )
            if self._folds and total_seconds > 0:
                average = total_seconds / max(fold_count, 1)
                best = min(self._folds, key=lambda item: item["seconds"])
                worst = max(self._folds, key=lambda item: item["seconds"])
                print(
                    "  · Average per fold: "
                    f"{average:.2f}s (fastest fold {int(best['fold'])}: {best['seconds']:.2f}s; "
                    f"slowest fold {int(worst['fold'])}: {worst['seconds']:.2f}s)"
                )
                if best.get("passes_per_second") and best.get("example_passes"):
                    print(
                        "  · Peak throughput: "
                        f"{best['passes_per_second']:,.1f} example-passes/s during fold {int(best['fold'])}"
                    )

        final_stage = self._sections.get("final_training")
        if final_stage:
            seconds = final_stage.get("seconds", 0.0)
            passes_value = final_stage.get("example_passes")
            epochs = final_stage.get("epochs")
            examples = final_stage.get("examples")
            if passes_value and seconds > 0:
                print(
                    f"- Final consolidation: {seconds:.2f}s for {passes_value:,.0f} example-passes "
                    f"({passes_value / seconds:,.1f} passes/s; epochs ~{epochs:.2f})"
                )
            else:
                print(f"- Final consolidation: {seconds:.2f}s (epochs ~{epochs or 0:.2f})")

        total_elapsed = self._total_elapsed
        if total_elapsed is not None:
            if self._total_passes > 0 and total_elapsed > 0:
                print(
                    f"- Overall runtime: {total_elapsed:.2f}s (~{self._total_passes:,.0f} example-passes "
                    f"at {self._total_passes / total_elapsed:,.1f}/s)"
                )
            else:
                print(f"- Overall runtime: {total_elapsed:.2f}s")
        if self._epoch_details:
            print("\nTraining epoch breakdown:")
            for entry in self._epoch_details:
                stage = entry.get("stage", "stage")
                epoch = int(entry.get("epoch", 0))
                seconds = entry.get("seconds", 0.0)
                examples = entry.get("examples", 0.0)
                tokens = entry.get("tokens", 0.0)
                batches = entry.get("batches", 0.0)
                throughput = (examples / seconds) if seconds > 0 and examples > 0 else 0.0
                token_throughput = (tokens / seconds) if seconds > 0 and tokens > 0 else 0.0
                print(
                    "- {stage} epoch {epoch}: {seconds:.2f}s; {examples:,.0f} examples "
                    "({throughput:,.1f}/s); {tokens:,.0f} tokens ({token_throughput:,.1f}/s); "
                    "{batches:,.0f} batches".format(
                        stage=stage,
                        epoch=epoch,
                        seconds=seconds,
                        examples=examples,
                        throughput=throughput,
                        tokens=tokens,
                        token_throughput=token_throughput,
                        batches=batches,
                    )
                )
        if self._training_elapsed > 0 and self._total_passes > 0:
            per_pass = self._training_elapsed / self._total_passes
            print(
                "- Observed training throughput: {passes:,.1f} passes in {seconds:.2f}s -> {rate:,.6f}s/pass".format(
                    passes=self._total_passes,
                    seconds=self._training_elapsed,
                    rate=per_pass,
                )
            )
        if self._total_tokens > 0 and self._training_elapsed > 0:
            per_token = self._training_elapsed / self._total_tokens
            print(
                "- Token throughput: {tokens:,.0f} tokens in {seconds:.2f}s -> {rate:,.6f}s/token ({per_sec:,.0f} tokens/s)".format(
                    tokens=self._total_tokens,
                    seconds=self._training_elapsed,
                    rate=per_token,
                    per_sec=(self._total_tokens / self._training_elapsed) if self._training_elapsed > 0 else 0.0,
                )
            )
        if self._complexity_notes:
            print("\nSpeed test complexity calibration:")
            for key in sorted(self._complexity_notes):
                value = self._complexity_notes[key]
                if isinstance(value, float):
                    if abs(value) >= 1.0:
                        display = f"{value:,.2f}"
                    elif value == 0.0:
                        display = "0"
                    elif abs(value) < 1e-4:
                        display = f"{value:.3e}"
                    else:
                        display = f"{value:.4f}"
                else:
                    display = str(value)
                print(f"- {key.replace('_', ' ')}: {display}")
        if self._calibration is not None:
            print(
                "- Calibration target: {examples:,} examples across {epochs:.2f} epoch(s) -> {seconds:.1f}s".format(
                    examples=self._calibration.examples,
                    epochs=self._calibration.epochs,
                    seconds=self._calibration.seconds,
                )
            )
        if self._estimates:
            print("\nProjected runtime estimates:")
            for estimate in self._estimates:
                label = estimate.get("label", "Projected runtime")
                seconds = float(estimate.get("estimated_seconds", 0.0))
                formatted = self._format_duration(seconds)
                target_examples = int(estimate.get("target_examples", 0))
                observed_examples = int(estimate.get("observed_examples", 0))
                multiplier = float(estimate.get("multiplier", 0.0))
                ratio_fragment = ""
                if observed_examples > 0:
                    ratio_fragment = (
                        f" ({target_examples:,} examples; {multiplier:.2f}× observed run of {observed_examples:,})"
                    )
                print(f"- {label}: ~{formatted}{ratio_fragment}")
                per_pass_detail = float(estimate.get("per_pass_seconds", 0.0))
                calibration_scale = float(estimate.get("calibration_scale", 0.0))
                per_token_detail = float(estimate.get("per_token_seconds", 0.0))
                tokens_per_pass_detail = float(estimate.get("tokens_per_pass", 0.0))
                tokens_per_pass_adjusted = float(estimate.get("tokens_per_pass_adjusted", 0.0))
                detail_bits: list[str] = []
                if per_pass_detail > 0:
                    source = estimate.get("pass_seconds_source", "baseline")
                    detail_bits.append(f"per-pass {per_pass_detail:.6f}s ({source})")
                if per_token_detail > 0 and tokens_per_pass_detail > 0:
                    detail_bits.append(
                        f"{tokens_per_pass_detail:,.1f} tokens/pass at {per_token_detail:.6f}s/token"
                    )
                token_scale = float(estimate.get("token_scale", 0.0))
                observed_avg_tokens = float(estimate.get("observed_average_tokens", 0.0))
                target_avg_tokens = float(estimate.get("target_average_tokens", 0.0))
                target_total_tokens = float(estimate.get("target_total_tokens", 0.0))
                if tokens_per_pass_adjusted > 0 and abs(tokens_per_pass_adjusted - tokens_per_pass_detail) > 1e-6:
                    detail_bits.append(
                        f"scaled tokens/pass {tokens_per_pass_adjusted:,.1f}"
                    )
                if target_avg_tokens > 0:
                    fragment = f"target avg tokens {target_avg_tokens:,.1f}"
                    if observed_avg_tokens > 0 and token_scale > 0:
                        fragment += f" (scale ×{token_scale:.3f} from {observed_avg_tokens:,.1f})"
                    detail_bits.append(fragment)
                if target_total_tokens > 0:
                    detail_bits.append(f"~{target_total_tokens:,.0f} target tokens")
                if calibration_scale > 0:
                    detail_bits.append(f"calibration ×{calibration_scale:.3f}")
                dataset_anchor_seconds = float(estimate.get("dataset_anchor_seconds", 0.0))
                dataset_anchor_key = estimate.get("dataset_anchor_key") or ""
                dataset_anchor_scale = float(estimate.get("dataset_anchor_scale", 0.0))
                dataset_anchor_per_pass = float(estimate.get("dataset_anchor_per_pass", 0.0))
                if dataset_anchor_seconds > 0:
                    fragment = f"anchor {dataset_anchor_seconds:.1f}s"
                    if dataset_anchor_key:
                        fragment += f" ({dataset_anchor_key})"
                    if dataset_anchor_per_pass > 0:
                        fragment += f" → {dataset_anchor_per_pass:.6f}s/pass"
                    detail_bits.append(fragment)
                if dataset_anchor_scale > 0 and dataset_anchor_seconds > 0:
                    detail_bits.append(f"anchor scale ×{dataset_anchor_scale:.3f}")
                if detail_bits:
                    print("  ↳ " + "; ".join(detail_bits))
                candidate_rows = cast(list[Mapping[str, float]], estimate.get("per_pass_candidates") or [])
                if candidate_rows:
                    print("  ↳ per-pass blend components:")
                    for candidate in candidate_rows:
                        source = candidate.get("source", "candidate")
                        seconds_candidate = float(candidate.get("seconds", 0.0))
                        weight_candidate = float(candidate.get("weight", 0.0))
                        confidence_candidate = float(candidate.get("confidence", 0.0))
                        evidence_candidate = float(candidate.get("evidence", 0.0))
                        adjusted_candidate = float(candidate.get("adjusted_weight", 0.0))
                        jitter_candidate = float(candidate.get("jitter", 0.0))
                        fragment = (
                            "    · {source}: {seconds:.6f}s (w {weight:.2f}, conf {confidence:.2f}, evid {evidence:.1f}".format(
                                source=source,
                                seconds=seconds_candidate,
                                weight=weight_candidate,
                                confidence=confidence_candidate,
                                evidence=evidence_candidate,
                            )
                        )
                        if adjusted_candidate > 0:
                            fragment += f", adj {adjusted_candidate:.2f}"
                        if jitter_candidate:
                            fragment += f", jitter {jitter_candidate:+.3f}"
                        fragment += ")"
                        print(fragment)
                blend_notes = cast(Mapping[str, float], estimate.get("blend_notes") or {})
                if blend_notes:
                    summary_parts: list[str] = []

                    def _append_blend(label: str, key: str, fmt: str) -> None:
                        value = blend_notes.get(key)
                        if value is None:
                            return
                        summary_parts.append(f"{label}={fmt.format(value)}")

                    _append_blend("arith", "blend_arithmetic", "{:.6f}s")
                    _append_blend("geom", "blend_geometric", "{:.6f}s")
                    _append_blend("harm", "blend_harmonic", "{:.6f}s")
                    _append_blend("quad", "blend_quadratic", "{:.6f}s")
                    _append_blend("median", "blend_median", "{:.6f}s")
                    _append_blend("trimmed", "blend_trimmed_mean", "{:.6f}s")
                    _append_blend("min", "blend_min", "{:.6f}s")
                    _append_blend("max", "blend_max", "{:.6f}s")
                    _append_blend("std", "blend_std", "{:.6f}s")
                    _append_blend("cv", "blend_cv", "{:.4f}")
                    _append_blend("stability", "blend_stability", "{:.4f}")
                    _append_blend("conf", "blend_confidence", "{:.4f}")
                    _append_blend("evidence", "blend_evidence", "{:.2f}")
                    _append_blend("entropy", "blend_entropy", "{:.4f}")
                    _append_blend("entropy_ratio", "blend_entropy_ratio", "{:.4f}")
                    _append_blend("weight", "blend_weight_sum", "{:.2f}")
                    _append_blend("components", "blend_component_count", "{:.0f}")
                    _append_blend("anchor_pull", "blend_anchor_pull", "{:.4f}")
                    if summary_parts:
                        print("    · blend stats: " + "; ".join(summary_parts))

                training_candidate_rows = cast(list[Mapping[str, float]], estimate.get("training_candidates") or [])
                if training_candidate_rows:
                    print("  ↳ training blend components:")
                    for candidate in training_candidate_rows:
                        source = candidate.get("source", "candidate")
                        seconds_candidate = float(candidate.get("seconds", 0.0))
                        weight_candidate = float(candidate.get("weight", 0.0))
                        confidence_candidate = float(candidate.get("confidence", 0.0))
                        evidence_candidate = float(candidate.get("evidence", 0.0))
                        adjusted_candidate = float(candidate.get("adjusted_weight", 0.0))
                        fragment = (
                            "    · {source}: {seconds:.2f}s (w {weight:.2f}, conf {confidence:.2f}, evid {evidence:.1f}".format(
                                source=source,
                                seconds=seconds_candidate,
                                weight=weight_candidate,
                                confidence=confidence_candidate,
                                evidence=evidence_candidate,
                            )
                        )
                        if adjusted_candidate > 0:
                            fragment += f", adj {adjusted_candidate:.2f}"
                        fragment += ")"
                        print(fragment)
                training_blend_notes = cast(Mapping[str, float], estimate.get("training_blend_notes") or {})
                if training_blend_notes:
                    summary_parts: list[str] = []

                    def _append_training(label: str, key: str, fmt: str) -> None:
                        value = training_blend_notes.get(key)
                        if value is None:
                            return
                        summary_parts.append(f"{label}={fmt.format(value)}")

                    _append_training("arith", "blend_arithmetic", "{:.2f}s")
                    _append_training("geom", "blend_geometric", "{:.2f}s")
                    _append_training("harm", "blend_harmonic", "{:.2f}s")
                    _append_training("median", "blend_median", "{:.2f}s")
                    _append_training("std", "blend_std", "{:.2f}s")
                    _append_training("cv", "blend_cv", "{:.4f}")
                    _append_training("stability", "blend_stability", "{:.4f}")
                    _append_training("conf", "blend_confidence", "{:.4f}")
                    _append_training("evidence", "blend_evidence", "{:.2f}")
                    _append_training("entropy", "blend_entropy", "{:.4f}")
                    _append_training("anchor_pull", "blend_anchor_pull", "{:.4f}")
                    if summary_parts:
                        print("    · training blend stats: " + "; ".join(summary_parts))
        print()


def _attempt_optional_install(module_name: str, package_spec: str) -> bool:
    """Best-effort automatic installation for optional runtime dependencies."""

    if importlib.util.find_spec(module_name) is not None:
        return True
    sentinel = f"{module_name}:{package_spec}"
    if sentinel in _OPTIONAL_INSTALL_ATTEMPTS:
        return False
    _OPTIONAL_INSTALL_ATTEMPTS.add(sentinel)
    print(f"Attempting to install optional dependency '{package_spec}' for module '{module_name}' ...")
    env = os.environ.copy()
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec], env=env)
    except subprocess.CalledProcessError as exc:
        print(
            f"Automatic installation of {package_spec!s} failed with exit code {exc.returncode}. "
            "Install the package manually and rerun the trainer."
        )
        return False
    return importlib.util.find_spec(module_name) is not None


def _ensure_module_available(module_name: str, package_spec: str | None = None) -> bool:
    """Check that a module can be imported, installing it on demand if possible."""

    if importlib.util.find_spec(module_name) is not None:
        return True
    if package_spec is None:
        return False
    return _attempt_optional_install(module_name, package_spec)


def _missing_core_modules() -> list[str]:
    missing: list[str] = []
    for module_name in ("torch", "numpy"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def summarise_labelled_dataset(
    path: Path,
    *,
    sample_limit: int | None = None,
    tokeniser: Callable[[str], Sequence[str]] | None = None,
) -> DatasetSummary:
    """Return dataset statistics for runtime projection heuristics."""

    summary = DatasetSummary(path=str(path))
    token_total = 0.0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return summary
            fieldnames = reader.fieldnames or []
            header_lower = {name.lower(): name for name in fieldnames if name}
            if "text" not in header_lower or "label" not in header_lower:
                return summary
            text_key = header_lower["text"]
            label_key = header_lower["label"]
            limit = sample_limit if sample_limit is not None and sample_limit > 0 else None
            for row in reader:
                summary.scanned_rows += 1
                text_raw = row.get(text_key)
                label_raw = row.get(label_key)
                if text_raw is None or label_raw is None:
                    summary.empty_rows += 1
                    continue
                text = str(text_raw).strip()
                label = str(label_raw).strip()
                if not text or not label:
                    summary.empty_rows += 1
                    continue
                summary.examples += 1
                if limit is not None and summary.token_samples >= limit:
                    continue
                if tokeniser is None:
                    token_count = len(text.split())
                else:
                    token_count = len(tokeniser(text))
                token_total += float(token_count)
                summary.token_samples += 1
    except FileNotFoundError:
        return summary
    except OSError:
        return summary
    except csv.Error:
        return summary

    if summary.token_samples > 0:
        summary.sample_size = summary.token_samples
        summary.average_tokens = token_total / float(summary.token_samples)
        summary.total_tokens = summary.average_tokens * float(summary.examples)
        summary.truncated = summary.token_samples < summary.examples
    else:
        summary.sample_size = sample_limit or 0
        summary.average_tokens = 0.0
        summary.total_tokens = 0.0
        summary.truncated = False
    return summary


def count_labelled_examples(path: Path) -> int:
    """Count the number of valid (text, label) pairs in ``path`` without loading them."""

    return summarise_labelled_dataset(path).examples


def _probe_torch_cuda_status() -> dict[str, Any] | None:
    """Inspect the local PyTorch installation for CUDA capability."""

    spec = importlib.util.find_spec("torch")
    if spec is None:
        return None

    try:
        torch_mod = importlib.import_module("torch")
    except ModuleNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - defensive catch for exotic import errors
        return {"import_error": str(exc)}

    version_cuda = getattr(getattr(torch_mod, "version", None), "cuda", None)
    build_has_cuda = version_cuda is not None
    runtime_available = False
    runtime_error: str | None = None
    device_count: int | None = None

    if build_has_cuda:
        try:
            runtime_available = bool(torch_mod.cuda.is_available())
        except Exception as exc:  # pragma: no cover - depends on host CUDA runtime state
            runtime_error = str(exc)
        else:
            if runtime_available:
                try:
                    device_count = int(torch_mod.cuda.device_count())
                except Exception:  # pragma: no cover - best-effort diagnostic only
                    device_count = None

    info: dict[str, Any] = {
        "build_has_cuda": build_has_cuda,
        "runtime_available": runtime_available,
        "version_cuda": version_cuda,
    }
    if runtime_error is not None:
        info["runtime_error"] = runtime_error
    if device_count is not None:
        info["device_count"] = device_count
    return info


def _variant_for_driver(driver_major: int) -> str:
    """Map an NVIDIA driver branch to the most compatible CUDA wheel."""

    return "cu121" if driver_major >= 525 else "cu118"


def _supported_torch_wheel_tags() -> tuple[set[str], str | None]:
    """Collect the set of platform tags supported by the current interpreter."""

    try:
        from packaging import tags as packaging_tags  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - packaging is expected under pip, but guard anyway.
        return set(), f"packaging.tags unavailable ({exc})"

    return {
        f"{tag.interpreter}-{tag.abi}-{tag.platform}".lower()
        for tag in packaging_tags.sys_tags()
    }, None


def _extract_torch_wheel_filenames(index_html: str, variant: str) -> list[str]:
    """Return wheel filenames advertised by the PyTorch download index."""

    filenames: list[str] = []
    href_pattern = re.compile(r'href="([^"]+\.whl[^"]*)"', re.IGNORECASE)
    for match in href_pattern.finditer(index_html):
        href = urllib.parse.unquote(match.group(1))
        filename = posixpath.basename(href.split("#", 1)[0])
        if filename.lower().startswith("torch-") and f"+{variant}" in filename:
            filenames.append(filename)
    return filenames


def _select_best_torch_wheel(
    variant: str,
    filenames: Sequence[str],
    supported_tags: set[str],
) -> tuple[str | None, str | None]:
    """Pick the newest compatible wheel for the given CUDA variant."""

    try:
        from packaging.version import InvalidVersion, Version  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - packaging should be present alongside pip.
        return None, f"packaging.version unavailable ({exc})"

    wheel_pattern = re.compile(
        r"^torch-([\w\.\-]+)\+" + re.escape(variant) + r"-([^-]+)-([^-]+)-([^.]+)\.whl$",
        re.IGNORECASE,
    )

    best_final: tuple[Version, str] | None = None
    best_prerelease: tuple[Version, str] | None = None

    for filename in filenames:
        match = wheel_pattern.match(filename)
        if match is None:
            continue
        version_str, py_tag, abi_tag, platform_tag = match.groups()
        suffix = f"{py_tag}-{abi_tag}-{platform_tag}".lower()
        if suffix not in supported_tags:
            continue
        try:
            version_obj = Version(version_str)
        except InvalidVersion:
            continue
        candidate = (version_obj, version_str)
        if version_obj.is_prerelease:
            if best_prerelease is None or version_obj > best_prerelease[0]:
                best_prerelease = candidate
        else:
            if best_final is None or version_obj > best_final[0]:
                best_final = candidate

    selected = best_final or best_prerelease
    if selected is None:
        return None, "no compatible wheel"

    return f"torch=={selected[1]}+{variant}", None


def _resolve_torch_spec_for_variant_exact(
    variant: str,
    supported_tags: set[str],
) -> tuple[str | None, str | None, str | None]:
    env_flag = os.environ.get("SOLARUS_TORCH_ALLOW_NIGHTLY")
    if env_flag is None:
        allow_nightly = True
    else:
        lowered = env_flag.strip().lower()
        if lowered in _SETUP_BOOL_FALSE:
            allow_nightly = False
        else:
            allow_nightly = lowered in _SETUP_BOOL_TRUE
    base_urls = ["https://download.pytorch.org/whl"]
    if allow_nightly:
        base_urls.append("https://download.pytorch.org/whl/nightly")

    errors: list[str] = []
    for base in base_urls:
        index_url = f"{base.rstrip('/')}/{variant}/torch/"
        try:
            with urllib.request.urlopen(index_url, timeout=5) as response:
                html = response.read().decode("utf-8", "ignore")
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            errors.append(f"{index_url}: {exc}")
            continue

        filenames = _extract_torch_wheel_filenames(html, variant)
        if not filenames:
            errors.append(f"{index_url}: no wheel files advertised")
            continue

        spec, spec_error = _select_best_torch_wheel(variant, filenames, supported_tags)
        if spec:
            return spec, index_url.rsplit("/torch", 1)[0], None
        if spec_error:
            errors.append(f"{index_url}: {spec_error}")

    return None, None, "; ".join(errors) if errors else None


def _resolve_torch_spec_for_variant(
    variant: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve a CUDA wheel spec and index URL compatible with this interpreter.

    Returns (spec, index_url, resolved_variant, error_message).
    """

    variant = (variant or "").strip().lower()
    if not variant:
        return None, None, None, "empty CUDA variant"

    supported_tags, tags_error = _supported_torch_wheel_tags()
    if not supported_tags:
        return None, None, None, tags_error or "unable to determine supported wheel tags"

    candidates: list[str] = [variant]
    candidates.extend(v for v in _CUDA_VARIANT_FALLBACKS.get(variant, ()) if v not in candidates)
    errors: list[str] = []
    for candidate in candidates:
        spec, index_url, error = _resolve_torch_spec_for_variant_exact(candidate, supported_tags)
        if spec:
            return spec, index_url, candidate, None
        if error:
            errors.append(f"{candidate}: {error}")

    return None, None, None, "; ".join(errors) if errors else None


def _detect_cuda_variant_from_nvidia_smi() -> str | None:
    """Attempt to infer the CUDA wheel variant using nvidia-smi output."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    line = result.stdout.strip().splitlines()[0:1]
    if not line:
        return None
    match = re.match(r"(\d+)", line[0])
    if not match:
        return None

    driver_major = int(match.group(1))
    return _variant_for_driver(driver_major)


def _parse_windows_driver_major(driver_version: str) -> int | None:
    """Best-effort conversion from the WMI driver version to the NVIDIA branch."""

    parts = driver_version.split(".")
    if len(parts) < 4:
        return None
    try:
        branch = int(parts[2])
        build = int(parts[3])
    except ValueError:
        return None

    combined = branch * 10000 + build
    if combined <= 100000:
        return None

    # Windows reports versions like 31.0.15.3168 for driver 531.68.
    driver_value = (combined / 100.0) - 1000.0
    if driver_value <= 0:
        return None
    return int(driver_value)


def _detect_cuda_variant_from_wmic() -> str | None:
    """Use the Windows Management Instrumentation output as a fallback detector."""

    if platform.system().lower() != "windows":
        return None

    command = [
        "wmic",
        "path",
        "win32_VideoController",
        "get",
        "Name,DriverVersion",
        "/format:csv",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    lines[0] = lines[0].lstrip("\ufeff")

    try:
        rows = list(csv.DictReader(lines))
    except csv.Error:
        return None

    for row in rows:
        name = (row.get("Name") or "").strip()
        if not name or "nvidia" not in name.lower():
            continue
        driver_version = (row.get("DriverVersion") or "").strip()
        if driver_version:
            driver_major = _parse_windows_driver_major(driver_version)
            if driver_major is not None:
                return _variant_for_driver(driver_major)
        # Default to the most recent CUDA build when we cannot parse the version.
        return "cu121"
    return None


def _detect_cuda_variant_from_driver() -> str | None:
    """Guess an appropriate CUDA wheel variant from locally detectable signals."""

    variant = _detect_cuda_variant_from_nvidia_smi()
    if variant is not None:
        return variant

    variant = _detect_cuda_variant_from_wmic()
    if variant is not None:
        return variant

    return None


def _preferred_cuda_variant_hint() -> str:
    """Determine which CUDA-enabled torch wheel should be installed."""

    for key in ("SOLARUS_TORCH_VARIANT", _PREFERRED_CUDA_VARIANT_ENV):
        value = os.environ.get(key)
        if value:
            return value.strip()

    detected = _detect_cuda_variant_from_driver()
    if detected:
        return detected

    return "cu121"


def _torch_requires_cuda_refresh() -> tuple[bool, str | None, str | None]:
    """Decide whether setup.py should reinstall PyTorch with CUDA support."""

    status = _probe_torch_cuda_status()
    if not status:
        return False, None, None
    if status.get("build_has_cuda"):
        return False, None, None

    env_requires_cuda = _env_flag_active(os.environ.get(_REQUIRE_CUDA_ENV))
    cli_requires_cuda = False
    device_cli_requires_cuda = False
    argv = list(sys.argv)
    for idx, token in enumerate(argv):
        lowered = token.lower()
        if lowered == "--require-cuda":
            cli_requires_cuda = True
        elif lowered == "--device":
            if idx + 1 < len(argv) and argv[idx + 1].strip().lower().startswith("cuda"):
                device_cli_requires_cuda = True
        elif lowered.startswith("--device="):
            value = token.split("=", 1)[1].strip().lower()
            if value.startswith("cuda"):
                device_cli_requires_cuda = True

    require_flag = env_requires_cuda or cli_requires_cuda or device_cli_requires_cuda
    detected_variant = _detect_cuda_variant_from_driver()
    env_variant = os.environ.get("SOLARUS_TORCH_VARIANT")
    if not require_flag and detected_variant is None and not env_variant:
        return False, None, None

    variant_hint = env_variant or _preferred_cuda_variant_hint()

    torch_spec_hint = os.environ.get("SOLARUS_TORCH_SPEC")
    torch_index_hint = os.environ.get("SOLARUS_TORCH_INDEX_URL")
    if not torch_spec_hint and variant_hint:
        (
            torch_spec_hint,
            torch_index_hint,
            resolved_variant,
            resolution_error,
        ) = _resolve_torch_spec_for_variant(variant_hint)
        if torch_spec_hint:
            if resolved_variant and resolved_variant != variant_hint:
                print(
                    f"CUDA variant '{variant_hint}' is unavailable for this Python build; "
                    f"selecting '{resolved_variant}' instead and installing the matching GPU wheel."
                )
                variant_hint = resolved_variant
            os.environ.setdefault("SOLARUS_TORCH_SPEC", torch_spec_hint)
            if torch_index_hint:
                os.environ.setdefault("SOLARUS_TORCH_INDEX_URL", torch_index_hint)
            if resolved_variant:
                os.environ.setdefault("SOLARUS_TORCH_VARIANT", resolved_variant)
            os.environ[_EXPECT_CUDA_ENV] = "1"
        else:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            system_name = platform.system() or "unknown system"
            message = (
                f"Unable to locate a CUDA-enabled torch wheel for variant '{variant_hint}' "
                f"compatible with Python {python_version} on {system_name}."
            )
            if resolution_error:
                message += f" Details: {resolution_error}"
            if require_flag:
                raise SystemExit(
                    message
                    + " Install a supported CUDA-enabled PyTorch build manually or switch to a supported Python version."
                )
            print(message + " Falling back to CPU execution.")
            return False, None, None

    if require_flag:
        if cli_requires_cuda or device_cli_requires_cuda:
            reason = (
                "CUDA execution was requested but the installed PyTorch build lacks CUDA support; "
                "refreshing with a CUDA-enabled wheel"
            )
        else:
            reason = (
                "PyTorch is installed without CUDA support; refreshing with a CUDA-enabled wheel"
            )
    else:
        reason = (
            "Detected NVIDIA GPU hardware but the current PyTorch build lacks CUDA support; "
            "refreshing with a CUDA-enabled wheel"
        )
    if variant_hint:
        reason += f" (variant hint: {variant_hint})"
    if torch_spec_hint:
        reason += f" [target spec {torch_spec_hint}]"
    reason += "."
    return True, reason, variant_hint or None


def _ensure_setuptools_available(env: Mapping[str, str]) -> None:
    """Ensure ``setuptools`` is importable before invoking ``setup.py``.

    Some container images omit setuptools to save space.  When our bootstrap
    logic tries to run ``setup.py`` in such environments the import failure
    surfaces as ``ModuleNotFoundError: No module named 'setuptools'`` before
    dependencies can even be installed.  To keep the self-install experience
    smooth we proactively bootstrap pip (via ``ensurepip``) and install
    setuptools (plus ``wheel`` for good measure) before invoking ``setup.py``.
    """

    if importlib.util.find_spec("setuptools") is not None:
        return

    print("Setuptools is missing; bootstrapping it before running setup.py ...")

    try:
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"], env=env)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            "Failed to bootstrap pip via ensurepip before running setup.py. "
            "Install setuptools manually and rerun train_intent_classifier.py."
        ) from exc

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "setuptools", "wheel"],
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Automatic setuptools installation failed. "
            "Resolve the issue (e.g. ensure network/pip availability) and rerun the trainer."
        ) from exc

    if importlib.util.find_spec("setuptools") is None:
        raise SystemExit(
            "Setuptools installation was attempted but the module remains unavailable. "
            "Install it manually and rerun train_intent_classifier.py."
        )


def _running_inside_visual_studio() -> bool:
    """Detect Visual Studio / VS Code environments to avoid relaunch confusion."""

    env = os.environ
    if env.get("VSCODE_PID") or env.get("VSCODE_CWD"):
        return True
    if env.get("TERM_PROGRAM", "").lower() == "vscode":
        return True
    for marker in ("VisualStudioVersion", "VSINSTALLDIR", "VisualStudioEdition"):
        if marker in env:
            return True
    return False


def _ensure_local_installation() -> None:
    if _env_flag_active(os.environ.get(_SETUP_SENTINEL_ENV)):
        return

    force_setup = _env_flag_active(os.environ.get(_FORCE_SETUP_ENV))
    cuda_bootstrap_attempted = os.environ.get(_CUDA_BOOTSTRAP_SENTINEL_ENV) is not None

    if cuda_bootstrap_attempted and not force_setup:
        status = _probe_torch_cuda_status()
        if status and status.get("build_has_cuda"):
            os.environ.pop(_CUDA_BOOTSTRAP_SENTINEL_ENV, None)
            cuda_bootstrap_attempted = False
        elif status and not status.get("build_has_cuda"):
            variant_hint = _preferred_cuda_variant_hint()
            raise SystemExit(
                "PyTorch still lacks CUDA support after the automatic installation attempt. "
                "Install a CUDA-enabled torch wheel manually (for example by setting "
                f"SOLARUS_TORCH_VARIANT={variant_hint}) and rerun train_intent_classifier.py, "
                "or export SOLARUS_TRAINER_FORCE_SETUP=1 to retry the bootstrap."
            )

    missing_modules = _missing_core_modules()
    need_cuda_refresh, cuda_reason, cuda_variant_hint = _torch_requires_cuda_refresh()

    if not missing_modules and not force_setup and not need_cuda_refresh:
        return

    setup_path = _resolve_project_root() / "setup.py"
    if not setup_path.is_file():
        if missing_modules:
            raise SystemExit(
                "Required dependencies are missing ("
                + ", ".join(missing_modules)
                + ") and setup.py could not be located for automatic installation."
            )
        return

    env = os.environ.copy()
    env[_SETUP_SENTINEL_ENV] = "1"

    _ensure_setuptools_available(env)

    reasons: list[str] = []
    if missing_modules:
        reasons.append("missing " + ", ".join(missing_modules))
    if need_cuda_refresh:
        reasons.append(cuda_reason or "refreshing PyTorch with CUDA support")
    if force_setup and not missing_modules:
        reasons.append("setup forced via SOLARUS_TRAINER_FORCE_SETUP")
    reason = " and ".join(reasons) if reasons else "dependency refresh"
    print(f"Running {setup_path.name} to install dependencies ({reason}).")

    if need_cuda_refresh:
        torch_spec_hint = os.environ.get("SOLARUS_TORCH_SPEC")
        torch_index_hint = os.environ.get("SOLARUS_TORCH_INDEX_URL")
        if torch_spec_hint:
            env.setdefault("SOLARUS_TORCH_SPEC", torch_spec_hint)
        if torch_index_hint:
            env.setdefault("SOLARUS_TORCH_INDEX_URL", torch_index_hint)
        if "SOLARUS_TORCH_VARIANT" not in env and cuda_variant_hint:
            env.setdefault("SOLARUS_TORCH_VARIANT", cuda_variant_hint)
        if _EXPECT_CUDA_ENV in os.environ:
            env[_EXPECT_CUDA_ENV] = os.environ[_EXPECT_CUDA_ENV]
        env[_CUDA_BOOTSTRAP_SENTINEL_ENV] = "1"
        os.environ[_CUDA_BOOTSTRAP_SENTINEL_ENV] = "1"

    try:
        subprocess.check_call([sys.executable, str(setup_path)], env=env)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Automatic setup failed with exit code {exc.returncode}. "
            "Resolve the installation issue and rerun train_intent_classifier.py."
        ) from exc

    script_path = str(_this_file) if _this_file is not None else sys.argv[0]
    relaunch_args = [sys.executable, script_path]
    relaunch_args.extend(sys.argv[1:])
    if _running_inside_visual_studio():
        print(
            "Dependencies installed successfully; continuing without relaunch "
            "(Visual Studio environment detected)."
        )
        return
    print("Dependencies installed successfully; relaunching train_intent_classifier.py ...")
    os.execvpe(sys.executable, relaunch_args, env)


if __name__ == "__main__":
    _ensure_local_installation()


def _script_directory() -> Path:
    """Best-effort location of this script when executed via notebooks/cells."""

    if _this_file is not None:
        return _this_file.parent
    return Path.cwd().resolve()

try:  # Python 3.11+ exposes datetime.UTC; provide a fallback for older runtimes.
    from datetime import UTC as _utc_zone  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - compatibility shim for Python < 3.11
    _utc_zone = timezone.utc  # type: ignore[assignment]

UTC_ZONE = _utc_zone

try:
    import torch as _torch  # type: ignore[import-not-found]
except ModuleNotFoundError:
    torch = cast(Any, None)  # type: ignore[assignment]
    _torch_available = False

    if not TYPE_CHECKING:
        class _TorchTensorPlaceholder:  # type: ignore[too-many-ancestors]
            """Fallback Tensor stand-in when torch is unavailable."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - sentinel
                raise RuntimeError("Torch tensors are unavailable in this environment.")

            def detach(self) -> _TorchTensorPlaceholder:  # pragma: no cover - sentinel
                return self

            def to(self, *args: Any, **kwargs: Any) -> _TorchTensorPlaceholder:  # pragma: no cover
                return self

            def numel(self) -> int:  # pragma: no cover - sentinel
                return 0

            def dim(self) -> int:  # pragma: no cover - sentinel
                return 0

            def size(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - sentinel
                return 0

            def __getitem__(self, item: Any) -> _TorchTensorPlaceholder:  # pragma: no cover
                return self

            def cpu(self) -> _TorchTensorPlaceholder:  # pragma: no cover - sentinel
                return self

            def sum(self, *args: Any, **kwargs: Any) -> _TorchTensorPlaceholder:  # pragma: no cover
                return self

            def unsqueeze(self, *args: Any, **kwargs: Any) -> _TorchTensorPlaceholder:  # pragma: no cover
                return self

            @property
            def dtype(self) -> Any:  # pragma: no cover - sentinel
                return None

            @property
            def device(self) -> Any:  # pragma: no cover - sentinel
                return None

            def tolist(self) -> list[Any]:  # pragma: no cover - sentinel
                return []

            def item(self) -> float:  # pragma: no cover - sentinel
                return 0.0

        TorchTensorType = _TorchTensorPlaceholder
else:
    torch = cast(Any, _torch)
    _torch_available = True
    if not TYPE_CHECKING:
        TorchTensorType = _torch.Tensor


if not _torch_available:
    def _parse_fallback_arguments() -> argparse.Namespace:
        """Parse a subset of CLI flags so the fallback trainer mirrors the UX."""

        parser = argparse.ArgumentParser(
            description=(
                "Lightweight intent classifier fallback that runs when PyTorch is unavailable."
            )
        )
        parser.add_argument("--dataset", default="data/intent_dataset.csv")
        parser.add_argument("--output-dir", default="models")
        parser.add_argument("--experiment-name", default="fallback_intent_classifier")
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--folds", type=int, default=1)
        parser.add_argument("--final-train-epochs", type=int, default=0)
        parser.add_argument("--distill-epochs", type=int, default=0)
        parser.add_argument("--self-train-rounds", type=int, default=0)
        parser.add_argument("--self-play-rounds", type=int, default=0)
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--embedding-dim", type=int, default=128)
        parser.add_argument("--hidden-dim", type=int, default=256)
        parser.add_argument("--ffn-dim", type=int, default=512)
        parser.add_argument("--encoder-layers", type=int, default=2)
        parser.add_argument("--attention-heads", type=int, default=4)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--dataloader-workers", type=int, default=0)
        parser.add_argument("--hardware-monitor-interval", type=float, default=5.0)
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--seed", type=int, default=13)
        parser.add_argument("--output-metrics", default=None)
        parser.add_argument("--no-keyword-calibration", action="store_true")
        parser.add_argument("--no-cognitive-router", action="store_true")
        parser.add_argument("--skip-overdrive-simulations", action="store_true")
        parser.add_argument("--no-performance-overdrive", action="store_true")
        parser.add_argument("--save-checkpoint", action="store_true")
        parser.add_argument("--checkpoint-name", default=None)
        parser.add_argument("--notes", default=None)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument(
            "--speed-test",
            action="store_true",
            help="Report dataset loading and training throughput statistics during the fallback run.",
        )
        parser.add_argument(
            "--estimate-dataset",
            default="data/intent_dataset.csv",
            help="Dataset path used when extrapolating runtime for the full training corpus.",
        )
        parser.add_argument(
            "--estimate-dataset-scan-limit",
            type=int,
            default=4096,
            help=(
                "Maximum labelled examples sampled when measuring token statistics for runtime projections."
            ),
        )
        parser.add_argument(
            "--speed-test-reference-gflops",
            type=float,
            default=0.0,
            help="Override the sustained GFLOP/s estimate used when projecting runtime.",
        )
        parser.add_argument(
            "--speed-test-calibration-seconds",
            type=float,
            default=0.0,
            help=(
                "Observed wall-clock seconds for one epoch on the target dataset; used to calibrate projections."
            ),
        )
        parser.add_argument(
            "--speed-test-calibration-examples",
            type=int,
            default=0,
            help="Number of labelled examples present in the calibration run.",
        )
        parser.add_argument(
            "--speed-test-calibration-epochs",
            type=float,
            default=1.0,
            help="Number of epochs completed during the calibration run (defaults to 1).",
        )
        known_args, unknown_args = parser.parse_known_args()
        if unknown_args:
            print(
                "[fallback trainer] Ignoring unsupported arguments: "
                + ", ".join(unknown_args)
            )
        known_args.estimate_dataset = Path(known_args.estimate_dataset).expanduser()
        return known_args


    @lru_cache(maxsize=65536)
    def _fallback_tokenise_cached(text: str) -> tuple[str, ...]:
        tokens = tuple(chunk for chunk in re.split(r"[^0-9A-Za-z']+", text.lower()) if chunk)
        if not tokens:
            return ("<blank>",)
        return tokens

    def _fallback_tokenise(text: str) -> tuple[str, ...]:
        """Lowercase tokeniser that keeps alphanumerics and apostrophes."""

        return _fallback_tokenise_cached(text)


    class _FallbackNaiveBayes:
        """Minimal multinomial naive Bayes classifier implemented without NumPy."""

        def __init__(self) -> None:
            self._label_counts: Counter[str] = Counter()
            self._token_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
            self._total_tokens: Counter[str] = Counter()
            self._vocabulary: set[str] = set()

        def fit(self, dataset: list[tuple[str, str]]) -> None:
            for text, label in dataset:
                self._label_counts[label] += 1
                tokens = _fallback_tokenise(text)
                for token in tokens:
                    self._token_counts[label][token] += 1
                    self._total_tokens[label] += 1
                    self._vocabulary.add(token)

        def predict(self, text: str) -> str:
            if not self._label_counts:
                return "unknown"
            vocab_size = max(1, len(self._vocabulary))
            total_examples = sum(self._label_counts.values())
            tokens = _fallback_tokenise(text)
            best_label = None
            best_score = float("-inf")
            for label, label_count in self._label_counts.items():
                log_prob = math.log((label_count + 1) / (total_examples + len(self._label_counts)))
                token_total = self._total_tokens[label]
                for token in tokens:
                    token_count = self._token_counts[label][token]
                    log_prob += math.log((token_count + 1) / (token_total + vocab_size))
                if log_prob > best_score:
                    best_score = log_prob
                    best_label = label
            return best_label or "unknown"

        def build_checkpoint_payload(self) -> dict[str, object]:
            """Expose serialisable model statistics without leaking internals."""

            return {
                "label_counts": dict(self._label_counts),
                "token_counts": {
                    label: dict(counter) for label, counter in self._token_counts.items()
                },
                "total_tokens": dict(self._total_tokens),
                "vocabulary": sorted(self._vocabulary),
            }


    def _fallback_split_folds(
        items: list[tuple[str, str]], folds: int
    ) -> list[tuple[list[tuple[str, str]], list[tuple[str, str]]]]:
        folds = max(1, folds)
        total = len(items)
        if total == 0:
            return []
        fold_slices: list[tuple[list[tuple[str, str]], list[tuple[str, str]]]] = []
        base_size = total // folds
        remainder = total % folds
        start = 0
        for fold_index in range(folds):
            fold_size = base_size + (1 if fold_index < remainder else 0)
            stop = start + fold_size
            val_items = items[start:stop]
            train_items = items[:start] + items[stop:]
            if not train_items:
                train_items = val_items
            if not val_items:
                val_items = train_items
            fold_slices.append((train_items, val_items))
            start = stop
        return fold_slices


    def _fallback_load_dataset(dataset_path: Path) -> list[tuple[str, str]]:
        if not dataset_path.exists():
            raise SystemExit(
                f"Dataset not found at {dataset_path}. Provide --dataset pointing to a CSV file."
            )
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows: list[tuple[str, str]] = []
            text_field = "text"
            label_field = "label"
            if reader.fieldnames is None:
                raise SystemExit(
                    "Fallback trainer expected a CSV header with at least 'text' and 'label' columns."
                )
            header_lower = {name.lower(): name for name in reader.fieldnames}
            if text_field not in header_lower:
                raise SystemExit("CSV is missing a 'text' column.")
            if label_field not in header_lower:
                raise SystemExit("CSV is missing a 'label' column.")
            text_key = header_lower[text_field]
            label_key = header_lower[label_field]
            for row in reader:
                text = row.get(text_key, "").strip()
                label = row.get(label_key, "").strip() or "unknown"
                if text:
                    rows.append((text, label))
            return rows


    def _fallback_evaluate(model: _FallbackNaiveBayes, items: list[tuple[str, str]]) -> float:
        if not items:
            return 0.0
        correct = 0
        for text, label in items:
            if model.predict(text) == label:
                correct += 1
        return correct / len(items)


    def _run_fallback_trainer() -> None:
        args = _parse_fallback_arguments()
        random.seed(args.seed)
        dataset_path = Path(args.dataset)
        speed_logger = SpeedTestLogger(args.speed_test)
        dataset_timer = speed_logger.marker()
        token_sample_limit = int(getattr(args, "estimate_dataset_scan_limit", 0) or 0)
        samples = _fallback_load_dataset(dataset_path)
        texts_only = [text for text, _ in samples]
        dataset_summary = _build_dataset_summary_from_texts(
            texts_only,
            path=dataset_path,
            sample_limit=token_sample_limit if token_sample_limit > 0 else None,
            tokeniser=_fallback_tokenise,
        )
        average_tokens = dataset_summary.average_tokens
        dataset_notes: dict[str, Any] = {
            "average_tokens": float(average_tokens),
            "token_samples": float(dataset_summary.token_samples),
            "estimated_tokens": float(dataset_summary.total_tokens),
        }
        if dataset_summary.examples > 0 and dataset_summary.token_samples > 0:
            dataset_notes["token_sample_fraction"] = (
                float(dataset_summary.token_samples) / float(dataset_summary.examples)
            )
        speed_logger.record_section(
            "dataset_load",
            dataset_timer,
            count=len(samples),
            notes=dataset_notes,
            add_to_total=False,
        )
        if not samples:
            print("No rows found in the dataset; nothing to train.")
            speed_logger.finish(total_examples=0)
            speed_logger.report()
            return
        random.shuffle(samples)
        folds = _fallback_split_folds(samples, args.folds)
        fold_metrics: list[dict[str, float]] = []
        if not folds:
            folds = [(samples, samples)]
        cv_timer = speed_logger.marker()
        for index, (train_items, val_items) in enumerate(folds, start=1):
            fold_timer = speed_logger.marker()
            model = _FallbackNaiveBayes()
            model.fit(train_items)
            accuracy = _fallback_evaluate(model, val_items)
            fold_metrics.append({"fold": index, "accuracy": accuracy})
            print(f"[fallback trainer] Fold {index}: accuracy={accuracy:.4f} ({len(val_items)} samples)")
            speed_logger.record_fold(
                index,
                fold_timer,
                examples=len(train_items),
                epochs=1.0,
            )
        speed_logger.record_section(
            "cross_validation",
            cv_timer,
            count=len(folds),
            add_to_total=False,
        )
        final_model = _FallbackNaiveBayes()
        final_timer = speed_logger.marker()
        final_model.fit(samples)
        speed_logger.record_section(
            "final_training",
            final_timer,
            passes=len(samples),
            notes={"epochs": 1.0, "examples": float(len(samples))},
        )
        final_accuracy = _fallback_evaluate(final_model, samples)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_prefix = args.experiment_name or "fallback_intent_classifier"
        metrics_payload = {
            "mode": "fallback",
            "folds": fold_metrics,
            "final_accuracy": final_accuracy,
            "samples": len(samples),
            "notes": args.notes,
        }
        metrics_path = output_dir / f"{artifact_prefix}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, indent=2)
        if args.output_metrics:
            with Path(args.output_metrics).open("w", encoding="utf-8") as handle:
                json.dump(metrics_payload, handle, indent=2)
        if args.save_checkpoint:
            checkpoint_name = args.checkpoint_name or f"{artifact_prefix}_model.json"
            checkpoint_path = output_dir / checkpoint_name
            checkpoint_payload = final_model.build_checkpoint_payload()
            with checkpoint_path.open("w", encoding="utf-8") as handle:
                json.dump(checkpoint_payload, handle, indent=2)
        speed_logger.finish(total_examples=len(samples))
        if speed_logger.enabled:
            profile = _build_speed_test_profile(
                args,
                average_tokens=average_tokens,
                reference_gflops=_resolve_reference_gflops(args),
                fallback_mode=True,
                observed_per_pass=speed_logger.baseline_per_pass(),
                observed_tokens_per_pass=speed_logger.average_tokens_per_pass(),
            )
            speed_logger.apply_complexity_profile(profile)
            calibration = _build_speed_test_calibration(args)
            if calibration is not None:
                speed_logger.configure_calibration(calibration)
            estimate_target = args.estimate_dataset
            try:
                if estimate_target.resolve(strict=False) == dataset_path.resolve(strict=False):
                    estimate_summary = dataset_summary
                else:
                    estimate_summary = summarise_labelled_dataset(
                        estimate_target,
                        sample_limit=token_sample_limit if token_sample_limit > 0 else None,
                        tokeniser=_fallback_tokenise,
                    )
            except OSError:
                estimate_summary = None
            estimated_examples = estimate_summary.examples if estimate_summary is not None else 0
            if estimated_examples > 0 and estimate_summary is not None:
                label = f"Projected runtime for {estimate_target.name}"
                speed_logger.register_estimate(
                    label,
                    estimated_examples,
                    observed_examples=len(samples),
                    target_average_tokens=estimate_summary.average_tokens,
                    target_total_tokens=estimate_summary.total_tokens,
                    observed_average_tokens=dataset_summary.average_tokens,
                    observed_total_tokens=dataset_summary.total_tokens,
                    dataset_summary=estimate_summary,
                    observed_dataset_summary=dataset_summary,
                )
            else:
                print(
                    f"[fallback trainer] Unable to project runtime: estimate dataset {estimate_target} is missing or empty."
                )
        speed_logger.report()
        print(
            "[fallback trainer] PyTorch not available; completed naive Bayes training with "
            f"final accuracy {final_accuracy:.4f} across {len(samples)} samples."
        )


    if __name__ == "__main__":
        _run_fallback_trainer()
        sys.exit(0)
    raise ImportError(
        "train_intent_classifier requires PyTorch when imported as a module. "
        "Install torch to access the full training pipeline."
    )

from torch import nn  # type: ignore[import-not-found]
from torch.nn import functional as F  # type: ignore[import-not-found]
from torch.utils.data import DataLoader, Dataset  # type: ignore[import-not-found]

adafactor_factory: Callable[..., TorchOptimizer] | None
try:
    from transformers.optimization import Adafactor as _Adafactor  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    _adafactor_available = False
    adafactor_factory = None
else:
    _adafactor_available = True
    adafactor_factory = cast(Callable[..., TorchOptimizer], _Adafactor)

_TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile")
from torch.nn.utils import clip_grad_norm_  # type: ignore[import-not-found]
from torch.optim import lr_scheduler  # type: ignore[import-not-found]
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn  # type: ignore[import-not-found]


def _build_amp_helpers():  # pragma: no cover - helper to keep AMP optional.
    try:  # Prefer the newer torch.amp API when available.
        from torch.amp import GradScaler as _GradScaler  # type: ignore[attr-defined]
        from torch.amp import autocast as _autocast  # type: ignore[attr-defined]

        def _create_scaler_amp(enabled: bool, device_type: str = "cuda"):
            if not enabled:
                return None
            try:
                return _GradScaler(device=device_type, enabled=True)
            except TypeError:
                return _GradScaler(enabled=True)
            except RuntimeError:
                if device_type != "cuda":
                    return _GradScaler(enabled=True)
                raise

        def _autocast_amp(enabled: bool, device_type: str = "cuda"):
            if not enabled:
                return contextlib.nullcontext()
            try:
                return _autocast(device_type=device_type, enabled=True)
            except TypeError:
                return _autocast(device_type="cuda", enabled=True)

        return _GradScaler, _create_scaler_amp, _autocast_amp
    except (ImportError, TypeError):
        try:
            from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore[attr-defined]
            from torch.cuda.amp import autocast as _autocast  # type: ignore[attr-defined]

            def _create_scaler_cuda(enabled: bool, device_type: str = "cuda"):
                if not enabled:
                    return None
                return _GradScaler(enabled=True)

            def _autocast_cuda(enabled: bool, device_type: str = "cuda"):
                return _autocast(enabled=enabled)

            return _GradScaler, _create_scaler_cuda, _autocast_cuda
        except ImportError:
            def _create_scaler_disabled(enabled: bool, device_type: str = "cuda"):
                return None

            def _null_autocast(enabled: bool, device_type: str = "cuda"):
                return contextlib.nullcontext()

            return None, _create_scaler_disabled, _null_autocast


class _GradScalerProtocol(Protocol):
    def is_enabled(self) -> bool: ...

    def scale(self, loss: TorchTensor) -> TorchTensor: ...

    def step(self, optimizer: TorchOptimizer) -> None: ...

    def update(self) -> None: ...

    def unscale_(self, optimizer: TorchOptimizer) -> None: ...


GradScaler, create_grad_scaler, autocast_context = _build_amp_helpers()


def _list_available_cuda_devices() -> list[tuple[TorchDevice, str]]:
    devices: list[tuple[TorchDevice, str]] = []
    if torch.cuda.is_available():
        try:
            count = torch.cuda.device_count()
        except Exception:
            count = 0
        for idx in range(max(0, count)):
            dev = torch.device(f"cuda:{idx}")
            try:
                name = torch.cuda.get_device_name(idx)
            except Exception:
                name = f"CUDA device {idx}"
            devices.append((dev, name))
    return devices


def _mps_backend_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        is_available = getattr(backend, "is_available", None)
        if callable(is_available):
            return bool(is_available())
        return False
    except Exception:
        return False


def _list_available_mps_devices() -> list[tuple[TorchDevice, str]]:
    if _mps_backend_available():
        return [(torch.device("mps"), "Apple Metal (MPS)")]
    return []


def _emit_cpu_bypass_diagnostics(
    device_info: Mapping[str, Any],
    fallback_reason: str | None,
) -> None:
    """Surface actionable guidance when CUDA could not be activated."""

    available = cast(dict[str, list[str]], device_info.get("available", {}))
    available_cuda = list(available.get("cuda", []))
    torch_cuda_available = torch.cuda.is_available()
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)

    if available_cuda:
        print(
            "PyTorch reported the following CUDA device(s), but they could not be "
            "initialised for training: " + ", ".join(available_cuda)
        )
    else:
        print(
            "PyTorch did not report any CUDA-capable GPU. Ensure the NVIDIA driver "
            "and CUDA runtime are installed."
        )

    print(
        "torch.cuda.is_available(): {available} | torch.version.cuda: {version}".format(
            available=torch_cuda_available,
            version=cuda_version if cuda_version is not None else "unknown",
        )
    )

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        print(f"CUDA_VISIBLE_DEVICES={visible_devices}")
    else:
        print("CUDA_VISIBLE_DEVICES is unset; all detected GPUs should be visible by default.")

    def _summarise(text: str) -> str:
        summary_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not summary_lines:
            return "(no output)"
        if len(summary_lines) > 3:
            return " | ".join(summary_lines[:3]) + " ..."
        return " | ".join(summary_lines)

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except FileNotFoundError:
        print(
            "nvidia-smi is not installed or not present in PATH; install the NVIDIA driver to access GPUs."
        )
    except subprocess.CalledProcessError as smi_error:
        stderr_summary = _summarise(smi_error.stderr or "")
        print(
            "nvidia-smi failed (exit code {code}). Output: {output}".format(
                code=smi_error.returncode,
                output=stderr_summary,
            )
        )
    except subprocess.TimeoutExpired:
        print("nvidia-smi timed out while probing the GPU; verify the driver installation.")
    else:
        stdout_summary = _summarise(smi.stdout)
        if stdout_summary != "(no output)":
            print("nvidia-smi detected GPU hardware: " + stdout_summary)
        else:
            print(
                "nvidia-smi executed successfully but did not list any GPUs. "
                "Ensure the GPU is connected and the driver is initialised."
            )

    if cuda_version is None:
        print(
            "Detected a CPU-only PyTorch build; install a CUDA-enabled torch wheel such as"
            " 'pip install torch --index-url https://download.pytorch.org/whl/cu118'."
        )

    if fallback_reason:
        print(f"CUDA initialisation failure details: {fallback_reason}")

    print(
        "Re-run the trainer on a CUDA-enabled machine without --allow-cpu-testing "
        "or --verify-device-only to execute full GPU training."
    )


def resolve_training_device(
    preference: str,
    *,
    allow_fallback: bool = True,
) -> tuple[TorchDevice, dict[str, Any]]:
    pref = (preference or "auto").strip().lower()
    cuda_devices = _list_available_cuda_devices()
    mps_devices = _list_available_mps_devices()
    cpu_device = torch.device("cpu")

    info: dict[str, Any] = {
        "requested": preference,
        "available": {},
    }
    if cuda_devices:
        info["available"]["cuda"] = [name for _dev, name in cuda_devices]
    if mps_devices:
        info["available"]["mps"] = [name for _dev, name in mps_devices]
    info["available"]["cpu"] = ["CPU"]

    def pick_cuda(index: int = 0) -> tuple[TorchDevice, str, int | None] | None:
        if not cuda_devices:
            return None
        clamped_index = max(0, min(index, len(cuda_devices) - 1))
        device_obj, name = cuda_devices[clamped_index]
        return device_obj, name, clamped_index

    def pick_mps() -> tuple[TorchDevice, str, int | None] | None:
        if not mps_devices:
            return None
        device_obj, name = mps_devices[0]
        return device_obj, name, None

    selected_device: TorchDevice | None = None
    selected_name: str | None = None
    selected_index: int | None = None
    selected_kind: str | None = None
    fallback_reason: str | None = None

    if pref in {"", "auto"}:
        candidate = pick_cuda()
        if candidate is not None:
            selected_device, selected_name, selected_index = candidate
            selected_kind = "cuda"
        else:
            candidate = pick_mps()
            if candidate is not None:
                selected_device, selected_name, selected_index = candidate
                selected_kind = "mps"
            else:
                selected_device = cpu_device
                selected_name = "CPU"
                selected_kind = "cpu"
    elif pref.startswith("cuda"):
        if not cuda_devices:
            if getattr(getattr(torch, "version", None), "cuda", None) is None:
                fallback_reason = (
                    "CUDA was requested but this PyTorch build does not include CUDA support "
                    "(torch.version.cuda is None)"
                )
            else:
                fallback_reason = "CUDA requested but no CUDA-capable devices were detected"
        else:
            if pref == "cuda":
                candidate = pick_cuda()
                if candidate is not None:
                    selected_device, selected_name, selected_index = candidate
                    selected_kind = "cuda"
            else:
                try:
                    index_token = pref.split(":", 1)[1]
                    requested_index = int(index_token)
                except (IndexError, ValueError):
                    raise ValueError(
                        f"Invalid CUDA device specification '{preference}'. Use 'cuda' or 'cuda:<index>'."
                    ) from None
                candidate = pick_cuda(requested_index)
                if candidate is not None:
                    selected_device, selected_name, selected_index = candidate
                    selected_kind = "cuda"
                else:
                    fallback_reason = (
                        f"CUDA device index {requested_index} is out of range "
                        f"(detected {len(cuda_devices)} device(s))"
                    )
    elif pref == "mps":
        candidate = pick_mps()
        if candidate is not None:
            selected_device, selected_name, selected_index = candidate
            selected_kind = "mps"
        else:
            fallback_reason = "MPS requested but the Metal backend is unavailable"
    elif pref == "cpu":
        selected_device = cpu_device
        selected_name = "CPU"
        selected_kind = "cpu"
    else:
        raise ValueError(
            f"Unrecognised device specification '{preference}'. "
            "Use 'auto', 'cpu', 'cuda', 'cuda:<index>', or 'mps'."
        )

    if selected_device is None:
        if fallback_reason is None:
            fallback_reason = f"Unable to satisfy device request '{preference}'."
        if not allow_fallback:
            _emit_cpu_bypass_diagnostics(info, fallback_reason)
            raise RuntimeError(fallback_reason)
        selected_device = cpu_device
        selected_name = "CPU"
        selected_kind = "cpu"

    if fallback_reason is not None and selected_kind == "cpu":
        info["fallback"] = fallback_reason

    if selected_kind is None and selected_device is not None:
        selected_kind = selected_device.type
    info["kind"] = selected_kind
    info["name"] = selected_name
    info["index"] = selected_index
    info["device"] = str(selected_device)

    assert selected_device is not None
    return selected_device, info

TRAINER_VERSION = "orion-trainer-0.7"

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
DEFAULT_BATCH_SIZE = 256


def _system_memory_snapshot() -> tuple[int | None, int | None]:
    """Return total and available system memory in bytes when detectable."""

    total_bytes: int | None = None
    available_bytes: int | None = None

    try:  # Prefer psutil when it is available.
        import psutil  # type: ignore[import-not-found]

        try:
            stats = psutil.virtual_memory()
        except Exception:  # pragma: no cover - very defensive
            stats = None
        else:
            total_bytes = int(getattr(stats, "total", 0) or 0) or None
            available_bytes = int(getattr(stats, "available", 0) or 0) or None
    except Exception:  # pragma: no cover - psutil is optional
        stats = None  # type: ignore[assignment]

    if total_bytes is None:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, ValueError, OSError):  # pragma: no cover - platform specific
            pass
        else:
            try:
                total_bytes = int(page_size) * int(phys_pages)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                total_bytes = None

    if available_bytes is None:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        except (AttributeError, ValueError, OSError):  # pragma: no cover - platform specific
            pass
        else:
            try:
                available_bytes = int(page_size) * int(avail_pages)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                available_bytes = None

    return total_bytes, available_bytes


def _bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / (1024.0 ** 3)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _safe_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:  # pragma: no cover - defensive
            return None
    try:
        floatable = cast(SupportsFloat, value)
        return float(floatable)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _probe_cuda_memory(device: TorchDevice) -> tuple[int | None, int | None]:
    """Attempt to capture free/total memory (in bytes) for ``device``."""

    if device.type != "cuda":
        return None, None
    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        return None, None

    free_bytes: int | None = None
    total_bytes: int | None = None

    try:
        with torch.cuda.device(device):
            free, total = torch.cuda.mem_get_info()
    except Exception:  # pragma: no cover - defensive against backend quirks
        free = total = None
    else:
        try:
            free_bytes = int(free)
            total_bytes = int(total)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            free_bytes = total_bytes = None

    if (free_bytes is None or total_bytes is None) and pynvml is not None:
        try:
            index = device.index if device.index is not None else torch.cuda.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        except Exception:  # pragma: no cover - NVML may be unavailable or fail
            pass
        else:
            try:
                free_bytes = int(getattr(info, "free", 0) or 0)
                total_bytes = int(getattr(info, "total", 0) or 0)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                free_bytes = total_bytes = None

    return free_bytes, total_bytes


def _enforce_cuda_memory_budget(
    args: argparse.Namespace,
    device: TorchDevice,
    device_info: dict[str, Any],
) -> tuple[TorchDevice, dict[str, Any]]:
    """Fallback to CPU when the selected CUDA device lacks free memory."""

    try:
        min_free_gb = float(getattr(args, "cuda_min_free_gb", 0.0) or 0.0)
    except (TypeError, ValueError):  # pragma: no cover - defensive parsing
        min_free_gb = 0.0
    if min_free_gb < 0:
        min_free_gb = 0.0

    args.cuda_memory_threshold_gb = min_free_gb
    args.cuda_memory_total_gb = None
    args.cuda_memory_free_gb = None
    args.cuda_memory_guard_enabled = min_free_gb > 0
    args.cuda_memory_guard_triggered = False
    args.cuda_memory_guard_forced = False
    args.cuda_memory_fallback_reason = None

    if device.type != "cuda":
        return device, device_info
    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        return device, device_info

    free_bytes, total_bytes = _probe_cuda_memory(device)
    if total_bytes is not None and total_bytes > 0:
        args.cuda_memory_total_gb = total_bytes / float(1024 ** 3)
    if free_bytes is not None and free_bytes >= 0:
        args.cuda_memory_free_gb = free_bytes / float(1024 ** 3)

    mem_bytes = device_info.setdefault("memory_bytes", {})
    if not isinstance(mem_bytes, dict):
        mem_bytes = {}
        device_info["memory_bytes"] = mem_bytes
    if free_bytes is not None:
        mem_bytes["free"] = int(free_bytes)
    if total_bytes is not None:
        mem_bytes["total"] = int(total_bytes)

    mem_gb = device_info.setdefault("memory_gb", {})
    if not isinstance(mem_gb, dict):
        mem_gb = {}
        device_info["memory_gb"] = mem_gb
    if args.cuda_memory_free_gb is not None:
        mem_gb["free"] = args.cuda_memory_free_gb
    if args.cuda_memory_total_gb is not None:
        mem_gb["total"] = args.cuda_memory_total_gb

    if not args.cuda_memory_guard_enabled or free_bytes is None:
        return device, device_info

    threshold_bytes = int(min_free_gb * (1024 ** 3))
    if free_bytes >= threshold_bytes:
        return device, device_info

    args.cuda_memory_guard_triggered = True
    free_gb = args.cuda_memory_free_gb if args.cuda_memory_free_gb is not None else free_bytes / float(1024 ** 3)
    detail = f"{free_gb:.2f} GiB free (requires ≥ {min_free_gb:.2f} GiB)"

    if getattr(args, "require_cuda", False):
        args.cuda_memory_guard_forced = True
        print(f"CUDA memory guard: {detail}. Continuing because --require-cuda is set.")
        return device, device_info

    args.cuda_memory_fallback_reason = detail
    print(
        "CUDA memory guard: {detail}. Falling back to CPU execution. "
        "Use --cuda-min-free-gb 0 to disable the guard or --require-cuda to override.".format(
            detail=detail
        )
    )
    try:
        torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - defensive cleanup
        pass

    fallback_reason = detail
    existing_reason = device_info.get("fallback")
    if isinstance(existing_reason, str) and existing_reason:
        fallback_reason = f"{existing_reason}; {detail}"
    device = torch.device("cpu")
    device_info["fallback"] = fallback_reason
    device_info["kind"] = "cpu"
    device_info["name"] = "CPU"
    device_info["index"] = None
    device_info["device"] = str(device)

    return device, device_info


def _run_cuda_warmup(device: TorchDevice, *, steps: int = 2) -> None:
    """Execute a few lightweight CUDA kernels so autotuners settle before training."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize(device)
    except Exception:  # pragma: no cover - sync may fail on exotic backends
        pass
    warmup_shape = (512, 512)
    tensor_a: TorchTensor | None = None
    tensor_b: TorchTensor | None = None
    for _ in range(max(1, steps)):
        tensor_a = torch.randn(warmup_shape, device=device, dtype=torch.float16)
        tensor_b = torch.randn(warmup_shape, device=device, dtype=torch.float16)
        _ = torch.matmul(tensor_a, tensor_b)
    try:
        torch.cuda.synchronize(device)
    except Exception:  # pragma: no cover - sync may fail on exotic backends
        pass
    finally:
        if tensor_a is not None:
            del tensor_a
        if tensor_b is not None:
            del tensor_b


def _new_adjustment_records() -> list[dict[str, Any]]:
    return []


def _new_str_list() -> list[str]:
    return []


def _new_float_list() -> list[float]:
    return []


@dataclass
class VramBudgetRegulator:
    """Dynamically shrink (or cautiously grow) micro-batches to respect VRAM limits."""

    device: TorchDevice
    target_free_gb: float = 3.5
    max_utilization: float = 0.95
    min_micro_batch: int = 16
    max_micro_batch: int = 512
    shrink_factor: float = 0.7
    grow_factor: float = 1.25
    shrink_cooldown: int = 3
    grow_cooldown: int = 16
    hysteresis_gb: float = 0.75

    _current_micro_batch: int | None = None
    _last_shrink_step: int = -1
    _last_grow_step: int = -1
    _adjustments: list[dict[str, Any]] = field(default_factory=_new_adjustment_records)

    def micro_batch_size_for(self, batch_size: int, suggested: int) -> int:
        """Return the micro-batch to use for this batch (initialising lazily)."""

        if batch_size <= 0:
            return max(1, suggested)
        candidate = max(1, min(batch_size, suggested))
        if self._current_micro_batch is None:
            seed = max(self.min_micro_batch, min(candidate, self.max_micro_batch))
            self._current_micro_batch = seed
        else:
            self._current_micro_batch = max(
                self.min_micro_batch,
                min(self._current_micro_batch, self.max_micro_batch, batch_size),
            )
            candidate = min(candidate, self._current_micro_batch)
        return max(1, min(batch_size, candidate, self._current_micro_batch))

    def observe(
        self,
        *,
        step_index: int,
        suggested: int,
    ) -> dict[str, Any] | None:
        """Inspect free VRAM and adjust the micro-batch if pressure is detected."""

        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = _probe_cuda_memory(self.device)
        if free_bytes is None or total_bytes is None or total_bytes <= 0:
            return None
        free_gb = free_bytes / float(1024 ** 3)
        total_gb = total_bytes / float(1024 ** 3)
        configured_limit = float(max(0.0, min(self.max_utilization, 1.0)))
        minimum_free_ratio = max(0.0, 1.0 - configured_limit)
        if total_gb > 0 and self.target_free_gb > 0:
            minimum_free_ratio = max(
                minimum_free_ratio,
                min(1.0, self.target_free_gb / total_gb),
            )
        minimum_free_ratio = min(minimum_free_ratio, 1.0)
        hysteresis_ratio = 0.0
        if total_gb > 0 and self.hysteresis_gb > 0:
            hysteresis_ratio = max(0.0, min(1.0, self.hysteresis_gb / total_gb))
        if self._current_micro_batch is None:
            self._current_micro_batch = max(
                self.min_micro_batch,
                min(suggested, self.max_micro_batch),
            )

        adjusted = False
        reason = ""
        free_ratio = min(max(free_gb / total_gb, 0.0), 1.0)
        utilisation = max(0.0, min(1.0, 1.0 - free_ratio))
        utilisation_limit = max(0.0, min(1.0, 1.0 - minimum_free_ratio))
        grow_ratio_threshold = min(1.0, minimum_free_ratio + hysteresis_ratio)
        if (
            free_ratio < minimum_free_ratio
            and step_index - self._last_shrink_step >= self.shrink_cooldown
        ):
            new_size = max(
                self.min_micro_batch,
                int(math.ceil(self._current_micro_batch * self.shrink_factor)),
            )
            if new_size < self._current_micro_batch:
                self._current_micro_batch = new_size
                self._last_shrink_step = step_index
                adjusted = True
                reason = "shrink"
        elif (
            free_ratio > grow_ratio_threshold
            and (self._current_micro_batch < min(self.max_micro_batch, suggested))
            and step_index - self._last_grow_step >= self.grow_cooldown
        ):
            new_size = min(
                min(self.max_micro_batch, suggested),
                int(math.ceil(self._current_micro_batch * self.grow_factor)),
            )
            if new_size > self._current_micro_batch:
                self._current_micro_batch = new_size
                self._last_grow_step = step_index
                adjusted = True
                reason = "grow"

        if adjusted:
            record: dict[str, float | str] = {
                "step": float(step_index),
                "free_gb": float(free_gb),
                "micro_batch": float(self._current_micro_batch),
                "utilization": float(utilisation),
                "utilization_limit": float(utilisation_limit),
                "configured_max_utilization": float(configured_limit),
            }
            if reason:
                record["action"] = reason
            self._adjustments.append(record)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return {
                "micro_batch": int(self._current_micro_batch),
                "action": reason or "adjust",
                "free_gb": float(free_gb),
                "utilization": float(utilisation),
                "utilization_limit": float(utilisation_limit),
                "configured_max_utilization": float(configured_limit),
            }
        return None

    def export(self) -> dict[str, Any]:
        return {
            "target_free_gb": float(self.target_free_gb),
            "max_utilization": float(self.max_utilization),
            "min_micro_batch": int(self.min_micro_batch),
            "max_micro_batch": int(self.max_micro_batch),
            "current_micro_batch": int(self._current_micro_batch) if self._current_micro_batch else None,
            "adjustments": list(self._adjustments),
        }


@dataclass
class CudaMemorySweeper:
    """Periodically release cached CUDA allocations when pressure builds up."""

    device: TorchDevice
    trigger_reserved_gb: float = 1.25
    trigger_free_gb: float = 2.5
    interval: int = 24
    also_collect_cpu: bool = True

    _last_sweep_step: int = 0
    _sweeps: list[dict[str, Any]] = field(default_factory=_new_adjustment_records)

    def maybe_sweep(self, step_index: int) -> dict[str, Any] | None:
        """Release cached allocations when reserved memory grows too large."""

        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        if self.trigger_reserved_gb <= 0 and self.trigger_free_gb <= 0:
            return None
        if step_index - self._last_sweep_step < max(1, self.interval):
            return None
        try:
            reserved_bytes = torch.cuda.memory_reserved(self.device)
            allocated_bytes = torch.cuda.memory_allocated(self.device)
        except Exception:
            return None
        reserved_gb = reserved_bytes / float(1024 ** 3)
        allocated_gb = allocated_bytes / float(1024 ** 3)
        free_bytes, total_bytes = _probe_cuda_memory(self.device)
        free_gb = free_bytes / float(1024 ** 3) if free_bytes is not None else None
        total_gb = total_bytes / float(1024 ** 3) if total_bytes is not None else None
        should_trigger = reserved_gb >= max(self.trigger_reserved_gb, 0.0)
        if not should_trigger and free_gb is not None and self.trigger_free_gb > 0:
            should_trigger = free_gb <= self.trigger_free_gb
        if not should_trigger:
            return None
        start_time = time.perf_counter()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        if self.also_collect_cpu:
            gc.collect()
        sweep_duration = time.perf_counter() - start_time
        free_after_bytes, _ = _probe_cuda_memory(self.device)
        free_after_gb = (
            free_after_bytes / float(1024 ** 3)
            if free_after_bytes is not None
            else None
        )
        record: dict[str, Any] = {
            "step": float(step_index),
            "reserved_gb": float(reserved_gb),
            "allocated_gb": float(allocated_gb),
            "free_gb_before": float(free_gb) if free_gb is not None else float("nan"),
            "free_gb_after": float(free_after_gb) if free_after_gb is not None else float("nan"),
            "duration": float(sweep_duration),
        }
        if total_gb is not None:
            record["total_gb"] = float(total_gb)
        self._sweeps.append(record)
        self._last_sweep_step = step_index
        return record

    def export(self) -> dict[str, Any]:
        return {
            "trigger_reserved_gb": float(self.trigger_reserved_gb),
            "trigger_free_gb": float(self.trigger_free_gb),
            "interval": int(self.interval),
            "sweeps": list(self._sweeps),
        }


@dataclass
class NvidiaTelemetrySample:
    """Single telemetry reading captured from ``nvidia-smi`` or NVML."""

    timestamp: float
    power_watts: float | None
    sm_clock_mhz: float | None
    mem_clock_mhz: float | None
    memory_util_percent: float | None
    temperature_c: float | None
    fan_percent: float | None
    fan_rpm: float | None


def _summarise_numeric_series(values: Sequence[float]) -> dict[str, Any]:
    """Produce a small descriptive summary for a numeric series."""

    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
        "last": float(values[-1]),
    }


class NvidiaSmiMonitor:
    """Continuously sample GPU telemetry using ``nvidia-smi``."""

    def __init__(
        self,
        gpu_index: int,
        *,
        binary: str = "nvidia-smi",
        interval: float = 0.5,
        max_samples: int = 4096,
    ) -> None:
        self.binary = binary
        self.gpu_index = gpu_index
        self.sample_interval = max(0.1, float(interval))
        self.samples: deque[NvidiaTelemetrySample] = deque(maxlen=max_samples)
        self._errors: deque[str] = deque(maxlen=32)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._query_fields: str | None = None
        self._field_tokens: list[str] = []
        self._collects_rpm = False
        self._nvml_handle = None
        self._nvml_error_recorded = False
        self.available = shutil.which(binary) is not None
        self.reason: str | None = None
        if self.available:
            self._initialise_query()
        else:
            self.reason = "binary_not_found"

    def _initialise_query(self) -> None:
        candidates = [
            "power.draw,clocks.sm,clocks.mem,utilization.memory,temperature.gpu,fan.speed,fan.speed.rpm",
            "power.draw,clocks.sm,clocks.mem,utilization.memory,temperature.gpu,fan.speed",
        ]
        for fields in candidates:
            result = self._execute_query(fields)
            if result is not None:
                self._query_fields = fields
                self._field_tokens = [token.strip() for token in fields.split(",")]
                self._collects_rpm = "fan.speed.rpm" in self._field_tokens
                if _nvml_loaded and pynvml is not None:
                    try:
                        self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                    except Exception:
                        self._nvml_handle = None
                return
        self.available = False
        self.reason = "query_failed"

    def start(self) -> bool:
        if not self.available:
            return False
        if self._thread is not None:
            return True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"nvidia-smi-monitor-{self.gpu_index}",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.sample_interval * 3.0))
        self._thread = None

    def _execute_query(self, fields: str) -> str | None:
        cmd = [
            self.binary,
            f"-i={self.gpu_index}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
        except FileNotFoundError:
            self.available = False
            self.reason = "binary_not_found"
            return None
        except subprocess.TimeoutExpired:
            self._record_error("query_timeout")
            return None
        except subprocess.CalledProcessError as error:
            self._record_error(f"query_failed:{error.returncode}")
            return None
        output = (completed.stdout or "").strip().splitlines()
        if not output:
            return None
        return output[-1]

    def _parse_float(self, token: str) -> float | None:
        text = token.strip()
        if not text or text in {"N/A", "nan", "None"}:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _record_error(self, message: str) -> None:
        with self._lock:
            self._errors.append(message)

    def _augment_with_nvml(self, fan_percent: float | None, fan_rpm: float | None) -> tuple[float | None, float | None]:
        if fan_rpm is not None and fan_percent is not None:
            return fan_percent, fan_rpm
        if self._nvml_handle is None or not _nvml_loaded or pynvml is None:
            return fan_percent, fan_rpm
        try:
            if hasattr(pynvml, "nvmlDeviceGetFanSpeed_v2"):
                fan_data = pynvml.nvmlDeviceGetFanSpeed_v2(self._nvml_handle, 0)
                rpm_value = getattr(fan_data, "rpm", None)
                percent_value = getattr(fan_data, "percent", None)
                if rpm_value not in (None, getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", None)):
                    fan_rpm = _safe_float(rpm_value)
                if percent_value not in (None, getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", None)):
                    if fan_percent is None:
                        fan_percent = _safe_float(percent_value)
            else:
                percent_value = pynvml.nvmlDeviceGetFanSpeed(self._nvml_handle)
                if percent_value not in (None, getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", None)):
                    if fan_percent is None:
                        fan_percent = _safe_float(percent_value)
        except Exception as error:
            if not self._nvml_error_recorded:
                self._record_error(f"nvml:{error}")
                self._nvml_error_recorded = True
        return fan_percent, fan_rpm

    def _collect_sample(self) -> NvidiaTelemetrySample | None:
        if not self.available or self._query_fields is None:
            return None
        line = self._execute_query(self._query_fields)
        if line is None:
            return None
        tokens = [token.strip() for token in line.split(",")]
        if len(tokens) != len(self._field_tokens):
            self._record_error("malformed_output")
            return None
        parsed = {name: self._parse_float(value) for name, value in zip(self._field_tokens, tokens)}
        fan_percent = parsed.get("fan.speed")
        fan_rpm = parsed.get("fan.speed.rpm") if self._collects_rpm else None
        fan_percent, fan_rpm = self._augment_with_nvml(fan_percent, fan_rpm)
        return NvidiaTelemetrySample(
            timestamp=time.time(),
            power_watts=parsed.get("power.draw"),
            sm_clock_mhz=parsed.get("clocks.sm"),
            mem_clock_mhz=parsed.get("clocks.mem"),
            memory_util_percent=parsed.get("utilization.memory"),
            temperature_c=parsed.get("temperature.gpu"),
            fan_percent=fan_percent,
            fan_rpm=fan_rpm,
        )

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._collect_sample()
            if sample is not None:
                with self._lock:
                    self.samples.append(sample)
            self._stop_event.wait(self.sample_interval)

    def summary(self) -> dict[str, Any]:
        if not self.available:
            return {
                "backend": "nvidia-smi",
                "enabled": False,
                "reason": self.reason or "unavailable",
                "device_index": self.gpu_index,
                "interval_s": self.sample_interval,
                "samples": 0,
            }
        with self._lock:
            samples = list(self.samples)
            errors = list(self._errors)
        summary: dict[str, Any] = {
            "backend": "nvidia-smi",
            "enabled": True,
            "device_index": self.gpu_index,
            "interval_s": self.sample_interval,
            "samples": len(samples),
        }
        if errors:
            summary["errors"] = errors

        def _pack(attribute: str) -> dict[str, Any] | None:
            values = [
                getattr(sample, attribute)
                for sample in samples
                if getattr(sample, attribute) is not None
            ]
            if not values:
                return None
            return _summarise_numeric_series(cast(Sequence[float], values))

        summary["power_watts"] = _pack("power_watts")
        summary["sm_clock_mhz"] = _pack("sm_clock_mhz")
        summary["mem_clock_mhz"] = _pack("mem_clock_mhz")
        summary["memory_util_percent"] = _pack("memory_util_percent")
        summary["temperature_c"] = _pack("temperature_c")
        summary["fan_percent"] = _pack("fan_percent")
        summary["fan_rpm"] = _pack("fan_rpm")
        if summary.get("samples", 0) == 0 and self.reason:
            summary["reason"] = self.reason
        return summary


class HardwareMonitorController:
    """Wrapper that manages GPU telemetry collection and exposes snapshots."""

    def __init__(
        self,
        device_kind: str,
        *,
        device_index: int | None,
        binary: str = "nvidia-smi",
        interval: float = 0.5,
    ) -> None:
        self.device_kind = device_kind
        self.device_index = device_index
        self.interval = max(0.1, float(interval))
        self.binary = binary
        self._reason: str | None = None
        if device_kind == "cuda" and device_index is not None:
            monitor = NvidiaSmiMonitor(device_index, binary=binary, interval=self.interval)
            if monitor.available:
                self._monitor = monitor
            else:
                self._reason = monitor.reason
                self._monitor = None
        else:
            self._monitor = None
            self._reason = "unsupported_device"

    @property
    def available(self) -> bool:
        return self._monitor is not None

    @property
    def reason(self) -> str | None:
        if self._monitor is not None:
            return self._monitor.reason
        return self._reason

    def start(self) -> bool:
        if self._monitor is None:
            return False
        return self._monitor.start()

    def snapshot(self) -> dict[str, Any]:
        if self._monitor is None:
            return {
                "backend": "nvidia-smi",
                "enabled": False,
                "reason": self._reason or "not_available",
                "device_index": self.device_index,
                "interval_s": self.interval,
                "samples": 0,
            }
        return self._monitor.summary()

    def stop_and_summarise(self) -> dict[str, Any]:
        if self._monitor is None:
            return {
                "backend": "nvidia-smi",
                "enabled": False,
                "reason": self._reason or "not_available",
                "device_index": self.device_index,
                "interval_s": self.interval,
                "samples": 0,
            }
        self._monitor.stop()
        return self._monitor.summary()

def _apply_memory_guard(args: argparse.Namespace) -> None:
    """Inspect host RAM and soften training pressure when memory is scarce."""

    guard_enabled = bool(getattr(args, "memory_guard", True))
    total_bytes, available_bytes = _system_memory_snapshot()
    total_gb = _bytes_to_gib(total_bytes)
    available_gb = _bytes_to_gib(available_bytes)

    budget_total = _safe_float(getattr(args, "memory_budget_gb", 0.0)) or 0.0
    budget_available = _safe_float(getattr(args, "memory_guard_min_available_gb", 0.0)) or 0.0

    total_threshold = budget_total if budget_total > 0 else 14.0
    available_threshold = budget_available if budget_available > 0 else 7.0

    args.memory_guard_total_gb = total_gb
    args.memory_guard_available_gb = available_gb
    args.memory_guard_total_threshold_gb = total_threshold
    args.memory_guard_available_threshold_gb = available_threshold
    adjustments: list[str] = []
    args.memory_guard_adjustments = adjustments
    trigger_reasons: list[str] = []
    args.memory_guard_active = False

    if not guard_enabled:
        args.memory_guard_trigger_reasons = []
        return

    if total_gb is not None and total_gb <= total_threshold + 1e-9:
        trigger_reasons.append("total_ram")
    if available_gb is not None and available_gb <= available_threshold + 1e-9:
        trigger_reasons.append("available_ram")
    if budget_total > 0 and total_gb is None:
        trigger_reasons.append("budget_unknown_total")
    if budget_available > 0 and available_gb is None:
        trigger_reasons.append("budget_unknown_available")

    args.memory_guard_trigger_reasons = trigger_reasons

    if not trigger_reasons:
        return

    args.memory_guard_active = True

    def _label(value: Any, *, none_label: str = "auto") -> str:
        if value is None:
            return none_label
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    def _record_adjustment(label: str, before: Any, after: Any) -> None:
        before_label = _label(before)
        after_label = _label(after)
        if before_label == after_label:
            return
        adjustments.append(f"{label}:{before_label}->{after_label}")

    batch_before_value = getattr(args, "batch_size", None)
    batch_before: int | None
    try:
        batch_before = int(batch_before_value) if batch_before_value is not None else None
    except (TypeError, ValueError):
        batch_before = None
    if batch_before is not None and batch_before > 1:
        batch_after = max(1, batch_before // 2)
        if batch_after < batch_before:
            args.batch_size = batch_after
            _record_adjustment("batch_size", batch_before_value, batch_after)

    worker_before_value = getattr(args, "dataloader_workers", None)
    worker_before: int | None
    try:
        worker_before = (
            int(worker_before_value) if worker_before_value is not None else None
        )
    except (TypeError, ValueError):
        worker_before = None
    if worker_before is None or worker_before != 0:
        args.dataloader_workers = 0
        _record_adjustment("dataloader_workers", worker_before_value, 0)

    prefetch_before_value = getattr(args, "dataloader_prefetch", None)
    prefetch_before: int | None
    try:
        prefetch_before = (
            int(prefetch_before_value) if prefetch_before_value is not None else None
        )
    except (TypeError, ValueError):
        prefetch_before = None
    if prefetch_before is None or prefetch_before != 1:
        args.dataloader_prefetch = 1
        _record_adjustment("dataloader_prefetch", prefetch_before_value, 1)

    total_str = f"{total_gb:.1f} GiB" if total_gb is not None else "unknown"
    avail_str = f"{available_gb:.1f} GiB" if available_gb is not None else "unknown"
    print(
        "Memory guard: limited host RAM detected "
        f"(total {total_str}, available {avail_str}); applying conservative settings."
    )
    if trigger_reasons:
        reason_str = ", ".join(trigger_reasons)
        print(f"Memory guard triggers: {reason_str}.")
    if adjustments:
        print("Memory guard adjustments:")
        for change in adjustments:
            print(f"  - {change}")
    else:
        print("Memory guard active: no configuration changes were required.")


def _memory_guard_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Expose the memory-guard state for downstream metrics/metadata."""

    adjustments = list(getattr(args, "memory_guard_adjustments", []))
    reasons = list(getattr(args, "memory_guard_trigger_reasons", []))
    summary: dict[str, Any] = {
        "enabled": bool(getattr(args, "memory_guard", False)),
        "active": bool(getattr(args, "memory_guard_active", False)),
        "total_ram_gb": _safe_float(getattr(args, "memory_guard_total_gb", None)),
        "available_ram_gb": _safe_float(getattr(args, "memory_guard_available_gb", None)),
        "total_threshold_gb": _safe_float(getattr(args, "memory_guard_total_threshold_gb", None)),
        "available_threshold_gb": _safe_float(getattr(args, "memory_guard_available_threshold_gb", None)),
        "adjustments": adjustments,
        "reasons": reasons,
    }

    batch_value = getattr(args, "batch_size", None)
    try:
        batch_size = int(batch_value) if batch_value is not None else None
    except (TypeError, ValueError):
        batch_size = None
    if batch_size is not None:
        summary["batch_size"] = batch_size

    worker_value = getattr(args, "dataloader_workers", None)
    try:
        workers = int(worker_value) if worker_value is not None else None
    except (TypeError, ValueError):
        workers = None
    if workers is not None:
        summary["dataloader_workers"] = workers

    prefetch_value = getattr(args, "dataloader_prefetch", None)
    try:
        prefetch = int(prefetch_value) if prefetch_value is not None else None
    except (TypeError, ValueError):
        prefetch = None
    if prefetch is not None:
        summary["dataloader_prefetch"] = prefetch

    return summary


def _cuda_memory_guard_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Summarise the CUDA memory safeguard for metadata exports."""

    threshold = _safe_float(getattr(args, "cuda_memory_threshold_gb", None))
    free = _safe_float(getattr(args, "cuda_memory_free_gb", None))
    total = _safe_float(getattr(args, "cuda_memory_total_gb", None))
    return {
        "enabled": bool(threshold and threshold > 0),
        "threshold_gb": threshold,
        "free_gb": free,
        "total_gb": total,
        "triggered": bool(getattr(args, "cuda_memory_guard_triggered", False)),
        "forced_cuda": bool(getattr(args, "cuda_memory_guard_forced", False)),
        "fallback_reason": getattr(args, "cuda_memory_fallback_reason", None),
    }


def _set_process_cpu_affinity(cpu_total: int, adjustments: list[str]) -> None:
    """Attempt to bind the current process to every available CPU core."""

    if cpu_total <= 0:
        return
    if not hasattr(os, "sched_setaffinity"):
        return
    try:
        current_affinity = os.sched_getaffinity(0)
    except (AttributeError, NotImplementedError, OSError):  # pragma: no cover - platform specific
        return
    desired_affinity = set(range(cpu_total))
    if set(current_affinity) == desired_affinity:
        return
    try:
        os.sched_setaffinity(0, desired_affinity)
    except (AttributeError, NotImplementedError, OSError):  # pragma: no cover - platform specific
        return
    try:
        current_size = len(current_affinity)
    except TypeError:
        current_size = 0
    adjustments.append(f"cpu_affinity:{current_size}->{len(desired_affinity)}")


def _estimate_overdrive_batch_size(
    current_batch: int,
    dataset_size: int,
    total_memory_bytes: int | None,
) -> int:
    """Heuristically expand the batch size to saturate large accelerators."""

    if total_memory_bytes is None or total_memory_bytes <= 0:
        return current_batch
    mem_gib = float(total_memory_bytes) / (1024.0 ** 3)
    candidate = int(current_batch)
    thresholds = [
        (96.0, 512),
        (64.0, 512),
        (48.0, 512),
        (32.0, 512),
        (24.0, 448),
        (16.0, 384),
        (12.0, 320),
        (8.0, 256),
        (6.0, 224),
        (4.0, 192),
        (3.0, 160),
    ]
    for threshold, value in thresholds:
        if mem_gib >= threshold:
            candidate = max(candidate, value)
            break
    candidate = max(candidate, current_batch)
    candidate = int(min(candidate, 512))
    if dataset_size > 0:
        candidate = int(min(candidate, max(dataset_size, candidate)))
    # Round to the nearest multiple of eight to keep tensor shapes CUDA friendly.
    if candidate % 8:
        candidate = ((candidate // 8) + 1) * 8
    return candidate


class _CUDAGraphStepWrapper(nn.Module):
    """Call ``torch.compiler.cudagraph_mark_step_begin`` before each forward pass."""

    def __init__(self, module: nn.Module, marker: Callable[[], None]) -> None:
        super().__init__()
        self._marker = marker
        # ``nn.Module`` registers child modules via ``__setattr__``. Assigning
        # the wrapped module through the normal attribute path ensures the
        # module shows up in ``self._modules`` while still being directly
        # retrievable with ``object.__getattribute__`` to avoid recursive
        # lookups inside ``__getattr__``.
        self._wrapped_module = module

    def forward(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        self._marker()
        return self._wrapped_module(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError as original_error:
            try:
                wrapped = object.__getattribute__(self, "_wrapped_module")
            except AttributeError:
                # ``_wrapped_module`` has not been initialised yet, so surface
                # the original error.
                raise original_error

            if hasattr(wrapped, name):
                return getattr(wrapped, name)
            raise original_error

    def __dir__(self) -> list[str]:
        """Expose attributes from both the wrapper and the wrapped module."""

        wrapped_attrs: Iterable[str]
        try:
            wrapped_attrs = dir(object.__getattribute__(self, "_wrapped_module"))
        except AttributeError:
            wrapped_attrs = ()
        return sorted(set(super().__dir__()) | set(wrapped_attrs))


@dataclass
class CompilePreflightReport:
    """Structured information about a CUDA ``torch.compile`` validation pass."""

    success: bool
    iterations: int
    batch_size: int
    sequence_length: int | None = None
    feature_dim: int | None = None
    has_emotion_features: bool = False
    kwarg_names: list[str] = field(default_factory=_new_str_list)
    input_dtype: str | None = None
    target_dtype: str | None = None
    logits_shape: list[int] | None = None
    logits_dtype: str | None = None
    loss_history: list[float] = field(default_factory=_new_float_list)
    gradient_total_norm: float = 0.0
    gradient_max_norm: float = 0.0
    missing_gradients: list[str] = field(default_factory=_new_str_list)
    warnings: list[str] = field(default_factory=_new_str_list)
    peak_memory_mb: int | None = None
    selected_mode: str | None = None


def _generate_compile_preflight_payload(
    args: argparse.Namespace, device: TorchDevice
):
    """Create synthetic inputs to validate ``torch.compile`` + CUDA graph execution."""

    if device.type != "cuda":
        return None

    num_classes = int(getattr(args, "_preflight_num_classes", 0) or 0)
    if num_classes <= 0:
        return None

    batch_hint = int(getattr(args, "_preflight_batch_size", 0) or getattr(args, "batch_size", 0) or 0)
    batch_size = int(max(2, min(batch_hint or 2, 8)))

    encoder_type = str(getattr(args, "encoder_type", "bilstm") or "bilstm").lower()
    forward_kwargs: dict[str, Any] = {}
    metadata: dict[str, Any] = {
        "batch_size": batch_size,
        "encoder_type": encoder_type,
        "has_emotion_features": False,
    }

    if encoder_type == "st":
        dim = int(getattr(args, "_preflight_sentence_dim", 0) or 0)
        if dim <= 0:
            return None
        inputs = torch.randn(batch_size, dim, device=device, dtype=torch.float32)
        metadata["feature_dim"] = dim
    else:
        seq_len = int(getattr(args, "_preflight_max_seq_len", 0) or getattr(args, "max_seq_len", 0) or 0)
        if seq_len <= 0:
            seq_len = 32
        vocab_size = int(getattr(args, "_preflight_vocab_size", 0) or 0)
        if vocab_size <= 0:
            vocab_size = 30522
        inputs = torch.randint(
            0,
            vocab_size,
            (batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )
        attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
        forward_kwargs["attention_mask"] = attention_mask
        metadata["sequence_length"] = seq_len
        metadata["vocab_size"] = vocab_size

    emotion_dim = int(getattr(args, "_preflight_emotion_dim", 0) or 0)
    if emotion_dim > 0:
        emotion_features = torch.randn(batch_size, emotion_dim, device=device, dtype=torch.float32)
        forward_kwargs["emotion_features"] = emotion_features
        metadata["emotion_dim"] = emotion_dim
        metadata["has_emotion_features"] = True

    targets = torch.randint(0, num_classes, (batch_size,), device=device, dtype=torch.long)
    metadata["num_classes"] = num_classes

    return inputs, forward_kwargs, targets, metadata


def _clone_preflight_tensor(tensor: TorchTensor) -> TorchTensor:
    """Return a detached clone suitable for repeated CUDA graph invocations."""

    return tensor.detach().clone(memory_format=torch.contiguous_format)


def _summarise_preflight_gradients(model: nn.Module) -> tuple[float, float, list[str], list[str]]:
    """Measure gradient health after a backward pass."""

    total_norm_sq = 0.0
    max_norm = 0.0
    missing: list[str] = []
    nonfinite: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        if grad is None:
            missing.append(name)
            continue
        grad_data = grad.detach()
        if grad_data.numel() == 0:
            continue
        grad_norm = float(grad_data.norm().item())
        total_norm_sq += grad_norm * grad_norm
        if grad_norm > max_norm:
            max_norm = grad_norm
        if not torch.isfinite(grad_data).all():
            nonfinite.append(name)
    total_norm = math.sqrt(total_norm_sq) if total_norm_sq > 0.0 else 0.0
    return total_norm, max_norm, missing, nonfinite


def _run_compiled_model_preflight(
    model: nn.Module, args: argparse.Namespace, device: TorchDevice
) -> CompilePreflightReport | None:
    """Execute forward/backward cycles to ensure CUDA graph safety and stability."""

    payload = _generate_compile_preflight_payload(args, device)
    if payload is None:
        return None

    inputs, forward_kwargs, targets, metadata = payload
    criterion = nn.CrossEntropyLoss().to(device)
    kwarg_names = sorted(forward_kwargs.keys())

    previous_mode = model.training
    model.train()

    num_iterations = 3
    loss_history: list[float] = []
    grad_total_norm: float = 0.0
    grad_max_norm: float = 0.0
    missing_parameters: set[str] = set()
    warnings: list[str] = []
    logits_signature: tuple[tuple[int, ...], TorchDType] | None = None

    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    try:
        for _ in range(num_iterations):
            call_inputs = _clone_preflight_tensor(inputs)
            call_kwargs = {
                key: _clone_preflight_tensor(value)
                if isinstance(value, TorchTensor)
                else value
                for key, value in forward_kwargs.items()
            }
            model.zero_grad(set_to_none=True)
            outputs_raw: TorchTensor | tuple[object, ...] = model(call_inputs, **call_kwargs)
            logits: TorchTensor
            if isinstance(outputs_raw, TorchTensorType):
                logits = outputs_raw
            else:
                outputs_tuple: tuple[object, ...] = outputs_raw
                first_output = outputs_tuple[0]
                if not isinstance(first_output, TorchTensorType):
                    raise TypeError(
                        "Compiled intent model must return a torch.Tensor as its first output."
                    )
                logits = first_output
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Preflight detected non-finite logits from compiled model.")
            if logits.shape[0] != targets.shape[0]:
                effective_targets = targets[: logits.shape[0]]
            else:
                effective_targets = targets
            loss = criterion(logits, effective_targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Preflight detected a non-finite loss from compiled model.")
            loss_history.append(float(loss.detach().item()))
            loss.backward()
            grad_total_norm, grad_max_norm, missing, nonfinite = _summarise_preflight_gradients(model)
            if nonfinite:
                raise RuntimeError(
                    "Preflight detected non-finite gradients for parameters: "
                    + ", ".join(nonfinite)
                )
            missing_parameters.update(missing)
            if grad_total_norm == 0.0 and not missing_parameters:
                warnings.append("Preflight gradients were zero; verify the loss function and targets.")
            if logits_signature is None:
                logits_signature = (tuple(logits.shape), logits.dtype)
            else:
                if logits_signature[0] != tuple(logits.shape) or logits_signature[1] != logits.dtype:
                    raise RuntimeError(
                        "Preflight observed changing logits signature across iterations, which "
                        "breaks CUDA graph capture safety."
                    )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
    finally:
        if not previous_mode:
            model.eval()

    peak_memory_mb: int | None = None
    if device.type == "cuda":
        try:
            peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        except Exception:
            peak_memory_bytes = None
        if peak_memory_bytes:
            peak_memory_mb = int(peak_memory_bytes // (1024 * 1024))

    trimmed_missing = sorted(missing_parameters)
    if len(trimmed_missing) > 16:
        warnings.append(
            "Preflight truncated missing-gradient list for readability; inspect the report for details."
        )
        trimmed_missing = trimmed_missing[:16]

    if trimmed_missing:
        warnings.append(
            "Some parameters did not receive gradients during preflight: "
            + ", ".join(trimmed_missing)
        )

    logits_shape_list: list[int] | None = None
    logits_dtype_name: str | None = None
    if logits_signature is not None:
        logits_shape_list = list(logits_signature[0])
        logits_dtype_name = str(logits_signature[1])

    report = CompilePreflightReport(
        success=True,
        iterations=num_iterations,
        batch_size=int(metadata.get("batch_size", inputs.shape[0])),
        sequence_length=metadata.get("sequence_length"),
        feature_dim=metadata.get("feature_dim"),
        has_emotion_features=bool(metadata.get("has_emotion_features", False)),
        kwarg_names=kwarg_names,
        input_dtype=str(inputs.dtype),
        target_dtype=str(targets.dtype),
        logits_shape=logits_shape_list,
        logits_dtype=logits_dtype_name,
        loss_history=loss_history,
        gradient_total_norm=grad_total_norm,
        gradient_max_norm=grad_max_norm,
        missing_gradients=trimmed_missing,
        warnings=warnings,
        peak_memory_mb=peak_memory_mb,
    )

    return report


def _snapshot_cuda_environment(device: TorchDevice) -> dict[str, Any]:
    """Capture high-level CUDA environment metadata for diagnostics."""

    snapshot: dict[str, Any] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
    }

    if not torch.cuda.is_available():
        return snapshot

    index = device.index if device.index is not None else torch.cuda.current_device()
    snapshot["device_index"] = index
    try:
        snapshot["device_name"] = torch.cuda.get_device_name(index)
    except Exception as error:
        snapshot["device_name_error"] = str(error)
    try:
        properties = torch.cuda.get_device_properties(index)
    except Exception as error:
        snapshot["device_properties_error"] = str(error)
    else:
        snapshot.update(
            {
                "compute_capability": f"{properties.major}.{properties.minor}",
                "multi_processor_count": int(properties.multi_processor_count),
                "total_memory_mb": int(properties.total_memory // (1024 * 1024)),
                "supports_cudagraphs": bool(getattr(torch.cuda, "CUDAGraph", None)),
            }
        )

    compiler_ns = getattr(torch, "compiler", None)
    snapshot["supports_cudagraph_mark_step"] = bool(
        getattr(compiler_ns, "cudagraph_mark_step_begin", None)
    )

    try:
        snapshot["driver_version"] = torch.cuda.driver_version()
    except Exception:
        snapshot["driver_version"] = None

    try:
        snapshot["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        snapshot["cudnn_version"] = None

    return snapshot

def _finalise_model_for_training(
    model: nn.Module, args: argparse.Namespace, device: TorchDevice
) -> nn.Module:
    """Apply compilation and multi-GPU wrapping when performance overdrive is active."""

    model = model.to(device)
    original_model = model
    overdrive_active = bool(getattr(args, "performance_overdrive_active", False))
    adjustments_raw = getattr(args, "performance_overdrive_adjustments", None)
    adjustments_list: list[str] | None
    if isinstance(adjustments_raw, list):
        adjustments_source = cast(list[object], adjustments_raw)
        adjustments_list = [str(item) for item in adjustments_source]
    else:
        adjustments_list = None

    if device.type == "cuda":
        args.performance_overdrive_environment_snapshot = _snapshot_cuda_environment(device)
    else:
        args.performance_overdrive_environment_snapshot = {
            "device": str(device),
            "cuda_available": False,
        }

    args.performance_overdrive_compile_report = {
        "success": False,
        "skipped": True,
        "reason": "torch.compile not attempted",
    }

    last_compile_error: BaseException | None = None
    compile_success = False
    compiled_candidate: nn.Module | None = None
    selected_mode: str | None = None
    preflight_report: CompilePreflightReport | None = None
    marker_attached = False

    if device.type == "cuda" and overdrive_active:
        if (
            _TORCH_COMPILE_AVAILABLE
            and not getattr(args, "disable_torch_compile", False)
        ):
            for mode in ("max-autotune", "reduce-overhead", None):
                try:
                    if mode is None:
                        compiled_model = torch.compile(model)  # type: ignore[attr-defined]
                        mode_label = "default"
                    else:
                        compiled_model = torch.compile(  # type: ignore[attr-defined]
                            model,
                            mode=mode,
                        )
                        mode_label = mode
                except TypeError:
                    try:
                        compiled_model = torch.compile(model)  # type: ignore[attr-defined]
                    except Exception as exc:  # pragma: no cover - torch.compile optional
                        last_compile_error = exc
                        continue
                    else:
                        mode_label = "default"
                except Exception as exc:  # pragma: no cover - torch.compile optional
                    last_compile_error = exc
                    continue
                else:
                    compiled_candidate = compiled_model
                    selected_mode = mode_label
                    compile_success = True
                    break
            if not compile_success and last_compile_error is not None:
                args.performance_overdrive_compile_error = str(last_compile_error)

    marker_fn: Callable[[], None] | None = None
    compiler_ns = getattr(torch, "compiler", None)
    if compiler_ns is not None:
        marker_candidate = getattr(compiler_ns, "cudagraph_mark_step_begin", None)
        if callable(marker_candidate):
            marker_fn = cast(Callable[[], None], marker_candidate)
    args.performance_overdrive_cudagraph_marker = False

    if compile_success and compiled_candidate is not None:
        if marker_fn is not None and not isinstance(compiled_candidate, _CUDAGraphStepWrapper):
            compiled_candidate = _CUDAGraphStepWrapper(compiled_candidate, marker_fn)
            marker_attached = True
        else:
            marker_attached = False
            try:
                preflight_report = _run_compiled_model_preflight(compiled_candidate, args, device)
            except Exception as preflight_error:
                compile_success = False
                compiled_candidate = None
                selected_mode = None
                args.performance_overdrive_compile_error = str(preflight_error)
                args.performance_overdrive_compile_report = {
                    "success": False,
                    "skipped": False,
                    "error": str(preflight_error),
                }
                print(
                    "torch.compile preflight failed; reverting to eager execution."
                )
                if getattr(args, "debug", False):
                    print(f"Preflight error: {preflight_error}")
                model = original_model
            else:
                model = compiled_candidate
                args.performance_overdrive_compile_mode = selected_mode
                args.performance_overdrive_cudagraph_marker = marker_attached
                if preflight_report is not None:
                    preflight_report.selected_mode = selected_mode
                    args.performance_overdrive_compile_report = asdict(preflight_report)
                    if preflight_report.warnings and getattr(args, "debug", False):
                        print(
                            "Preflight warnings: " + "; ".join(preflight_report.warnings)
                        )
                else:
                    args.performance_overdrive_compile_report = {
                        "success": True,
                        "skipped": True,
                        "reason": "preflight skipped due to insufficient metadata",
                        "selected_mode": selected_mode,
                    }
                if (
                    adjustments_list is not None
                    and selected_mode is not None
                    and f"torch_compile:{selected_mode}" not in adjustments_list
                ):
                    adjustments_list.append(f"torch_compile:{selected_mode}")
                if preflight_report is not None and preflight_report.loss_history:
                    final_loss = preflight_report.loss_history[-1]
                    grad_norm = preflight_report.gradient_total_norm
                    print(
                        "torch.compile preflight succeeded; using compiled model. "
                        f"mode={selected_mode or 'default'}, loss={final_loss:.4f}, grad_norm={grad_norm:.4f}"
                    )
                else:
                    print("torch.compile preflight succeeded; using compiled model.")
        
        multi_gpu_devices = list(getattr(args, "performance_overdrive_multi_gpu_devices", []))
        if multi_gpu_devices and not isinstance(model, nn.DataParallel):
            base_index = device.index if device.index is not None else multi_gpu_devices[0]
            unique_devices = sorted({int(idx) for idx in multi_gpu_devices})
            if len(unique_devices) > 1:
                model = nn.DataParallel(
                    model,
                    device_ids=unique_devices,
                    output_device=base_index,
                )
                args.performance_overdrive_multi_gpu_wrapped = True
                if (
                    adjustments_list is not None
                    and f"multi_gpu:{len(unique_devices)}" not in adjustments_list
                ):
                    adjustments_list.append(f"multi_gpu:{len(unique_devices)}")

    return model


def _apply_performance_overdrive(
    args: argparse.Namespace,
    *,
    using_cuda: bool,
    using_mps: bool,
    dataset_size: int | None = None,
    cuda_diagnostics: Mapping[str, Any] | None = None,
    device: TorchDevice | None = None,
) -> None:
    """Dial every performance knob to maximise hardware utilisation."""

    enabled = bool(getattr(args, "performance_overdrive", True))
    adjustments: list[str] = []
    reasons: list[str] = []

    args.performance_overdrive_enabled = enabled
    args.performance_overdrive_adjustments = adjustments
    args.performance_overdrive_reasons = reasons
    args.performance_overdrive_active = False
    args.performance_overdrive_multi_gpu = False
    args.performance_overdrive_multi_gpu_devices = []
    args.performance_overdrive_multi_gpu_wrapped = False
    args.performance_overdrive_compile_mode = None
    args.performance_overdrive_compile_error = None
    args.performance_overdrive_target_batch = None
    args.performance_overdrive_float32_precision = None
    args.performance_overdrive_total_vram_gb = None

    if not enabled:
        reasons.append("disabled")
        return

    if getattr(args, "memory_guard_active", False):
        reasons.append("memory_guard")
        return

    args.performance_overdrive_active = True

    cpu_total = os.cpu_count() or 0
    if cpu_total > 0 and hasattr(torch, "set_num_threads"):
        current_threads = torch.get_num_threads() if hasattr(torch, "get_num_threads") else None
        target_threads = max(1, cpu_total)
        if current_threads is None or current_threads < target_threads:
            torch.set_num_threads(target_threads)
            before = current_threads if current_threads is not None else "auto"
            adjustments.append(f"torch_num_threads:{before}->{target_threads}")

    if cpu_total > 0 and hasattr(torch, "set_num_interop_threads"):
        current_interop = (
            torch.get_num_interop_threads() if hasattr(torch, "get_num_interop_threads") else None
        )
        target_interop = max(1, min(cpu_total, cpu_total // 2 or 1))
        if current_interop is None or current_interop < target_interop:
            torch.set_num_interop_threads(target_interop)
            before = current_interop if current_interop is not None else "auto"
            adjustments.append(f"torch_num_interop_threads:{before}->{target_interop}")

    _set_process_cpu_affinity(cpu_total, adjustments)

    flush_getter = getattr(torch, "get_flush_denormal", None)
    flush_state: bool | None
    if callable(flush_getter):
        try:
            result = flush_getter()
        except Exception:  # pragma: no cover - backend optional
            flush_state = None
        else:
            flush_state = result if isinstance(result, bool) else None
    else:
        flush_state = None
    if hasattr(torch, "set_flush_denormal"):
        if flush_state is False or flush_state is None:
            try:
                torch.set_flush_denormal(True)
            except RuntimeError:  # pragma: no cover - backend optional
                pass
            else:
                adjustments.append("flush_denormals:on")

    if hasattr(torch, "set_float32_matmul_precision"):
        previous_precision = None
        getter = getattr(torch, "get_float32_matmul_precision", None)
        try:
            previous_precision = getter() if callable(getter) else None
        except Exception:  # pragma: no cover - backend optional
            previous_precision = None
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:  # pragma: no cover - backend optional
            pass
        else:
            if previous_precision != "high":
                label = previous_precision or "default"
                adjustments.append(f"float32_matmul_precision:{label}->high")
                args.performance_overdrive_float32_precision = "high"

    if using_cuda and hasattr(torch.backends, "cudnn") and torch.backends.cudnn is not None:
        if torch.backends.cudnn.benchmark:
            torch.backends.cudnn.benchmark = False
            adjustments.append("cudnn_benchmark:off")
        if hasattr(torch.backends.cudnn, "allow_tf32") and not torch.backends.cudnn.allow_tf32:
            torch.backends.cudnn.allow_tf32 = True
            adjustments.append("cudnn_tf32:on")

    if using_cuda and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        matmul = torch.backends.cuda.matmul
        if hasattr(matmul, "allow_tf32") and not matmul.allow_tf32:
            matmul.allow_tf32 = True
            adjustments.append("cuda_matmul_tf32:on")
        if hasattr(matmul, "allow_fp16_reduced_precision_reduction") and not matmul.allow_fp16_reduced_precision_reduction:
            matmul.allow_fp16_reduced_precision_reduction = True
            adjustments.append("cuda_matmul_fp16_redux:on")
        if hasattr(matmul, "allow_bf16_reduced_precision_reduction") and not matmul.allow_bf16_reduced_precision_reduction:
            matmul.allow_bf16_reduced_precision_reduction = True
            adjustments.append("cuda_matmul_bf16_redux:on")

    if using_cuda and hasattr(torch.backends, "cuda"):
        sdp_kernel = getattr(torch.backends.cuda, "sdp_kernel", None)
        if sdp_kernel is not None:
            try:
                if hasattr(sdp_kernel, "is_flash_enabled") and not sdp_kernel.is_flash_enabled():
                    sdp_kernel.enable_flash_sdp(True)
                    adjustments.append("cuda_flash_sdp:on")
                if hasattr(sdp_kernel, "is_mem_efficient_enabled") and not sdp_kernel.is_mem_efficient_enabled():
                    sdp_kernel.enable_mem_efficient_sdp(True)
                    adjustments.append("cuda_mem_eff_sdp:on")
                if hasattr(sdp_kernel, "is_math_enabled") and not sdp_kernel.is_math_enabled():
                    sdp_kernel.enable_math_sdp(True)
                    adjustments.append("cuda_math_sdp:on")
            except Exception:  # pragma: no cover - backend optional
                pass

    if using_cuda or using_mps:
        desired_workers = _auto_dataloader_workers(performance_overdrive=True)
    else:
        desired_workers = max(1, cpu_total - 1) if cpu_total > 1 else 0

    worker_before = getattr(args, "dataloader_workers", None)
    try:
        worker_before_int = int(worker_before) if worker_before is not None else None
    except (TypeError, ValueError):
        worker_before_int = None

    if desired_workers > 0:
        if worker_before_int is None or worker_before_int < desired_workers:
            before_label = worker_before if worker_before is not None else "auto"
            args.dataloader_workers = desired_workers
            adjustments.append(f"dataloader_workers:{before_label}->{desired_workers}")
    elif worker_before_int is None:
        args.dataloader_workers = 0

    desired_prefetch = 2 if (using_cuda or using_mps) else 1
    prefetch_before = getattr(args, "dataloader_prefetch", None)
    try:
        prefetch_before_int = int(prefetch_before) if prefetch_before is not None else None
    except (TypeError, ValueError):
        prefetch_before_int = None

    if desired_prefetch > 0:
        if prefetch_before_int is None or prefetch_before_int < desired_prefetch:
            before_label = prefetch_before if prefetch_before is not None else "auto"
            args.dataloader_prefetch = desired_prefetch
            adjustments.append(f"dataloader_prefetch:{before_label}->{desired_prefetch}")

    if using_cuda:
        prime_override = bool(getattr(args, "cuda_prefetch_prime_user_override", False))
        if not prime_override and getattr(args, "cuda_prefetch_prime", False):
            args.cuda_prefetch_prime = False
            adjustments.append("cuda_prefetch_prime:on->off")

    total_memory_bytes: int | None = None
    if using_cuda:
        if cuda_diagnostics is not None:
            total_memory_bytes = int(cuda_diagnostics.get("total_memory_bytes", 0) or 0)
        elif device is not None:
            try:
                properties = torch.cuda.get_device_properties(device.index or 0)
                total_memory_bytes = int(getattr(properties, "total_memory", 0))
            except Exception:  # pragma: no cover - diagnostic fallback
                total_memory_bytes = None
        if total_memory_bytes:
            args.performance_overdrive_total_vram_gb = (
                float(total_memory_bytes) / (1024.0 ** 3)
            )

        current_batch = int(getattr(args, "batch_size", 0) or 0)
        if current_batch <= 0:
            current_batch = DEFAULT_BATCH_SIZE
        target_batch = _estimate_overdrive_batch_size(
            current_batch,
            dataset_size or 0,
            total_memory_bytes,
        )
        if target_batch > current_batch:
            adjustments.append(f"batch_size:{current_batch}->{target_batch}")
            args.batch_size = target_batch
            args.performance_overdrive_target_batch = target_batch
            final_current = getattr(args, "final_train_batch_size", 0) or 0
            if final_current <= 0 or final_current < target_batch:
                before_label = final_current or "inherit"
                args.final_train_batch_size = target_batch
                adjustments.append(f"final_batch:{before_label}->{target_batch}")

        grad_steps = max(1, int(getattr(args, "grad_accumulation_steps", 1)))
        desired_steps = grad_steps
        if desired_steps < 4:
            desired_steps = 4
        if target_batch > 2048:
            desired_steps = max(desired_steps, 8)
        if desired_steps > 8:
            desired_steps = 8
        if desired_steps != grad_steps:
            args.grad_accumulation_steps = desired_steps
            adjustments.append(f"grad_accumulation:{grad_steps}->{desired_steps}")

        try:
            device_count = torch.cuda.device_count()
        except Exception:  # pragma: no cover - CUDA optional
            device_count = 1
        if device_count and device_count > 1:
            args.performance_overdrive_multi_gpu = True
            args.performance_overdrive_multi_gpu_devices = list(range(device_count))

    if adjustments:
        print("Performance overdrive: saturating available hardware resources for maximum throughput.")
        for change in adjustments:
            print(f"  - {change}")
    else:
        print("Performance overdrive: configuration already utilises all available hardware.")


def _performance_overdrive_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Expose the performance-overdrive state for downstream metrics/metadata."""

    adjustments = list(getattr(args, "performance_overdrive_adjustments", []))
    reasons = list(getattr(args, "performance_overdrive_reasons", []))
    worker_value = getattr(args, "dataloader_workers", None)
    try:
        workers = int(worker_value) if worker_value is not None else None
    except (TypeError, ValueError):
        workers = None
    prefetch_value = getattr(args, "dataloader_prefetch", None)
    try:
        prefetch = int(prefetch_value) if prefetch_value is not None else None
    except (TypeError, ValueError):
        prefetch = None
    summary: dict[str, Any] = {
        "enabled": bool(getattr(args, "performance_overdrive", False)),
        "active": bool(getattr(args, "performance_overdrive_active", False)),
        "adjustments": adjustments,
        "reasons": reasons,
    }
    if workers is not None:
        summary["dataloader_workers"] = workers
    if prefetch is not None:
        summary["dataloader_prefetch"] = prefetch
    target_batch = getattr(args, "performance_overdrive_target_batch", None)
    if target_batch is not None:
        summary["target_batch_size"] = int(target_batch)
    compile_mode = getattr(args, "performance_overdrive_compile_mode", None)
    if compile_mode is not None:
        summary["torch_compile_mode"] = compile_mode
    compile_error = getattr(args, "performance_overdrive_compile_error", None)
    if compile_error:
        summary["torch_compile_error"] = str(compile_error)
    multi_gpu_devices = list(getattr(args, "performance_overdrive_multi_gpu_devices", []))
    if multi_gpu_devices:
        summary["multi_gpu_devices"] = multi_gpu_devices
        summary["multi_gpu_wrapped"] = bool(
            getattr(args, "performance_overdrive_multi_gpu_wrapped", False)
        )
    float32_precision = getattr(args, "performance_overdrive_float32_precision", None)
    if float32_precision is not None:
        summary["float32_matmul_precision"] = float32_precision
    total_vram = getattr(args, "performance_overdrive_total_vram_gb", None)
    if total_vram is not None:
        summary["total_vram_gb"] = float(total_vram)
    if getattr(args, "cuda_prefetch_prime", None) is not None:
        summary["cuda_prefetch_prime"] = bool(args.cuda_prefetch_prime)
    return summary


def _run_overdrive_simulations(
    args: argparse.Namespace,
    *,
    device: TorchDevice,
    using_cuda: bool,
    using_mps: bool,
) -> dict[str, Any]:
    """Execute high-intensity warm-up simulations to stress the active device."""

    enabled = bool(getattr(args, "overdrive_simulate", False))
    summary: dict[str, Any] = {
        "enabled": False,
        "reason": "disabled" if not enabled else "unsupported_device",
        "rounds": int(getattr(args, "overdrive_simulation_rounds", 0) or 0),
        "matrix_size": int(getattr(args, "overdrive_simulation_matrix", 0) or 0),
        "batch": int(getattr(args, "overdrive_simulation_batch", 0) or 0),
        "device": str(device),
    }
    if not enabled:
        return summary

    rounds = max(0, int(getattr(args, "overdrive_simulation_rounds", 0) or 0))
    if rounds <= 0:
        summary["reason"] = "no_rounds"
        return summary

    matrix_size = max(128, int(getattr(args, "overdrive_simulation_matrix", 128)))
    batch = max(1, int(getattr(args, "overdrive_simulation_batch", 1)))
    dtype: TorchDType
    if using_cuda:
        dtype = torch.float16 if getattr(args, "amp", True) else torch.float32
    elif using_mps:
        dtype = torch.float32
    else:
        dtype = torch.float32

    summary.update(
        {
            "enabled": True,
            "reason": "executed",
            "rounds": rounds,
            "matrix_size": matrix_size,
            "batch": batch,
            "dtype": str(dtype),
            "results": [],
        }
    )

    gflops_values: list[float] = []
    wall_durations: list[float] = []
    peak_bytes: list[float] = []
    mps_module = getattr(torch, "mps", None)

    def _synchronise() -> None:
        if using_cuda and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif using_mps and mps_module is not None and hasattr(mps_module, "synchronize"):
            try:
                mps_module.synchronize()
            except Exception:
                pass

    try:
        for round_idx in range(1, rounds + 1):
            if using_cuda and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                start_all = time.perf_counter()
                a = torch.randn((batch, matrix_size, matrix_size), device=device, dtype=dtype)
                b = torch.randn((batch, matrix_size, matrix_size), device=device, dtype=dtype)
                _synchronise()
                compute_start = time.perf_counter()
                product = torch.bmm(a, b)
                _synchronise()
                compute_end = time.perf_counter()
            duration = max(0.0, compute_end - compute_start)
            wall = max(0.0, compute_end - start_all)
            flops = 2.0 * (matrix_size ** 3) * batch
            gflops = float(flops / max(duration, 1e-9) / 1e9)
            gflops_values.append(gflops)
            wall_durations.append(wall)
            entry: dict[str, Any] = {
                "round": round_idx,
                "compute_s": float(duration),
                "wall_s": float(wall),
                "gflops": gflops,
            }
            if using_cuda and torch.cuda.is_available():
                peak = float(torch.cuda.max_memory_allocated(device))
                entry["peak_memory_bytes"] = peak
                peak_bytes.append(peak)
            summary["results"].append(entry)
            print(
                "Overdrive simulation round {idx}/{total}: {gflops:.2f} GFLOP/s "
                "(matrix {matrix}, batch {batch}, compute {compute:.3f}s, wall {wall:.3f}s).".format(
                    idx=round_idx,
                    total=rounds,
                    gflops=gflops,
                    matrix=matrix_size,
                    batch=batch,
                    compute=duration,
                    wall=wall,
                )
            )
            del a, b, product
            if using_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    except RuntimeError as error:
        summary["enabled"] = False
        summary["reason"] = f"error:{error}"[:200]
        print(f"Overdrive simulations aborted: {error}")
        return summary

    if gflops_values:
        summary["mean_gflops"] = float(sum(gflops_values) / len(gflops_values))
        summary["max_gflops"] = float(max(gflops_values))
    if wall_durations:
        summary["mean_wall_s"] = float(sum(wall_durations) / len(wall_durations))
    if peak_bytes:
        summary["max_peak_memory_bytes"] = float(max(peak_bytes))
    return summary


def _auto_dataloader_workers(performance_overdrive: bool = False) -> int:
    """Use all but one CPU core for data loading to maximise throughput."""

    cpu_total = os.cpu_count() or 0
    if cpu_total <= 1:
        return 0
    worker_budget = max(1, cpu_total - 1)
    # Leave a little headroom for host-side processing to keep RAM usage in check.
    if worker_budget > 11:
        worker_budget = 11
    return worker_budget


def _gather_cuda_diagnostics(device: TorchDevice) -> dict[str, Any]:
    """Collect a comprehensive snapshot of the active CUDA device."""

    if device.type != "cuda":
        raise ValueError("CUDA diagnostics requested for a non-CUDA device")

    properties = torch.cuda.get_device_properties(device)
    capability = f"{properties.major}.{properties.minor}"
    runtime_version = getattr(torch.version, "cuda", None)
    driver_version = None
    if hasattr(torch.cuda, "driver_version"):
        try:
            driver_version = torch.cuda.driver_version()
        except Exception:
            driver_version = None

    diagnostics: dict[str, Any] = {
        "name": properties.name,
        "capability": capability,
        "total_memory_bytes": int(properties.total_memory),
        "multi_processor_count": int(getattr(properties, "multi_processor_count", 0)),
        "runtime_version": runtime_version,
        "driver_version": driver_version,
    }

    return diagnostics


def _format_cuda_driver_version(value: object) -> str | None:
    """Format the NVIDIA driver version into a human readable string."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        integer = int(value)
        major = integer // 1000
        minor = integer % 1000
        if minor % 10 == 0:
            minor = minor // 10
        return f"{major}.{minor:02d}".rstrip("0").rstrip(".")
    return str(value)


def progressive_mlp_hidden_dims(initial_dim: int, num_layers: int, expansion: float) -> list[int]:
    """Compute the hidden widths for a progressive MLP stack."""

    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    if initial_dim < 1:
        raise ValueError("initial_dim must be positive")
    if expansion < 1.0:
        raise ValueError("expansion must be >= 1.0")
    dims: list[int] = []
    current = int(initial_dim)
    for layer in range(num_layers):
        current = max(1, int(current))
        dims.append(current)
        if layer < num_layers - 1:
            next_dim = int(round(current * expansion))
            current = max(next_dim, initial_dim)
    return dims


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _is_colab_environment() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ or Path("/content").exists()


def _deduplicate_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _candidate_data_directories() -> list[tuple[Path, bool]]:
    script_dir = _script_directory()
    repo_data_dir = script_dir / "data"
    cwd = Path.cwd()
    candidates: list[tuple[Path, bool]] = [
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

    merged: dict[str, tuple[Path, bool]] = {}
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


def _colab_drive_mounts() -> list[Path]:
    drive_root = Path("/content/drive")
    mounts: list[Path] = []
    for name in ("MyDrive", "My Drive"):
        candidate = drive_root / name
        try:
            if candidate.exists():
                mounts.append(candidate)
        except OSError:
            continue
    shared_drives = drive_root / "Shared drives"
    try:
        if shared_drives.exists():
            for entry in shared_drives.iterdir():
                try:
                    if entry.is_dir():
                        mounts.append(entry)
                except OSError:
                    continue
    except OSError:
        pass
    return mounts


def _colab_workspace_root() -> Path:
    env_override = os.environ.get("SOLARUS_COLAB_WORKSPACE")
    if env_override:
        return Path(env_override).expanduser()
    return Path("/content/solarus_workspace")


def _colab_model_dir_candidates() -> list[Path]:
    candidates: list[Path] = []
    for mount in _colab_drive_mounts():
        base = mount / "Solarus"
        candidates.append(base / "models")
    candidates.append(Path("/content/solarus_models"))
    return candidates


def _mirror_into_directory(source: Path, target_dir: Path) -> Path | None:
    try:
        if not source.exists():
            return None
    except OSError:
        return None
    try:
        resolved = source.resolve()
    except OSError:
        resolved = source
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Warning: unable to prepare Colab workspace directory {target_dir}: {exc}")
        return None
    destination = target_dir / resolved.name
    try:
        if destination.exists():
            return destination
    except OSError:
        return None
    try:
        shutil.copy2(resolved, destination)
    except OSError as exc:
        print(f"Warning: failed to mirror {resolved} into {destination}: {exc}")
        return None
    else:
        print(f"Colab setup: mirrored {resolved} to {destination} for convenience.")
        return destination


def _apply_colab_environment_adjustments(
    args: argparse.Namespace,
    *,
    model_dir_user_override: bool,
) -> dict[str, Any] | None:
    if not _is_colab_environment():
        return None

    adjustments: dict[str, Any] = {}
    workspace_root = _colab_workspace_root()
    adjustments["workspace_root"] = str(workspace_root)
    try:
        workspace_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        adjustments["workspace_error"] = str(exc)
        print(f"Warning: unable to create Colab workspace at {workspace_root}: {exc}")

    dataset_copies: dict[str, str] = {}
    target_data_dir = workspace_root / "data"
    mirrored = _mirror_into_directory(args.dataset, target_data_dir)
    if mirrored is not None:
        dataset_copies["labelled"] = str(mirrored)
    unlabeled = getattr(args, "unlabeled_dataset", None)
    if unlabeled is not None:
        mirrored_unlabeled = _mirror_into_directory(unlabeled, target_data_dir)
        if mirrored_unlabeled is not None:
            dataset_copies["unlabeled"] = str(mirrored_unlabeled)
    if dataset_copies:
        adjustments["dataset_copies"] = dataset_copies

    original_model_dir = Path(args.model_dir).expanduser()
    selected_model_dir = original_model_dir
    adjustments["model_dir_original"] = str(original_model_dir)
    if not model_dir_user_override:
        for candidate in _colab_model_dir_candidates():
            parent = candidate.parent
            try:
                if not parent.exists():
                    parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            selected_model_dir = candidate
            break
        if selected_model_dir != original_model_dir:
            print(
                "Colab environment detected: storing training artefacts under "
                f"{selected_model_dir} (override --model-dir to change)."
            )
    try:
        selected_model_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Warning: unable to create model directory {selected_model_dir}: {exc}")
    args.model_dir = selected_model_dir
    adjustments["model_dir_resolved"] = str(selected_model_dir)
    adjustments["model_dir_overridden"] = selected_model_dir != original_model_dir
    args.colab_environment = True
    args.colab_adjustments = adjustments
    return adjustments


def _strip_ipykernel_arguments(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Remove the automatic connection arguments that Jupyter/Colab inject."""

    filtered: list[str] = []
    ignored: list[str] = []
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
) -> Path | None:
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

    csv_entries: list[Path] = []
    for entry in entries:
        try:
            if entry.is_file():
                csv_entries.append(entry)
        except OSError:
            continue

    pattern_matches: list[Path] = []
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
    flag: str | None = None,
    search_names: Sequence[str] | None = None,
) -> Path:
    expanded = path.expanduser()
    original_target = expanded.resolve(strict=False)

    script_dir = _script_directory()
    candidates = _deduplicate_paths([
        expanded,
        script_dir / expanded,
    ])

    checked: list[Path] = []
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

    names: list[str] = []
    seen_names: set[str] = set()
    for candidate_name in [*(search_names or []), expanded.name]:
        if candidate_name not in seen_names:
            seen_names.add(candidate_name)
            names.append(candidate_name)
    patterns: list[str] = []
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

    def best_accuracy(self) -> float | None:
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

    def create_run_directory(self, val_accuracy: float, tag: str | None = None) -> Path:
        timestamp = datetime.now(UTC_ZONE).strftime("%Y%m%dT%H%M%SZ")
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
        transformers_module = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - optional dependency
        if _attempt_optional_install("transformers", "transformers>=4.34"):
            transformers_module = importlib.import_module("transformers")
        else:
            raise ImportError(
                "The 'transformers' package is required for the transformer encoder. "
                "Install it via 'pip install transformers'."
            ) from exc
    AutoTokenizer = getattr(transformers_module, "AutoTokenizer")
    transformers_logging = getattr(transformers_module, "logging")
    transformers_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def load_sentence_transformer(model_name: str):
    try:
        st_module = importlib.import_module("sentence_transformers")
    except ImportError as exc:  # pragma: no cover - optional dependency
        if _attempt_optional_install("sentence_transformers", "sentence-transformers>=2.2.2"):
            st_module = importlib.import_module("sentence_transformers")
        else:
            raise ImportError(
                "The 'sentence-transformers' package is required for the st encoder. "
                "Install it via 'pip install sentence-transformers'."
            ) from exc
    SentenceTransformer = getattr(st_module, "SentenceTransformer")
    return SentenceTransformer(model_name)


def _resolve_encoder_choice(encoder: str) -> tuple[str, str | None]:
    """Pick a workable encoder given the available optional dependencies."""

    requested = encoder.strip().lower()
    reason: str | None = None

    if requested == "st":
        if not _ensure_module_available("sentence_transformers", "sentence-transformers>=2.2.2"):
            reason = "sentence-transformers dependency is missing"
            requested = "bilstm"
        else:
            try:
                importlib.import_module("sentence_transformers")
            except Exception as exc:  # pragma: no cover - defensive guard
                message = str(exc).splitlines()[0]
                reason = f"failed to initialise sentence-transformers ({message})"
                requested = "bilstm"

    if requested == "transformer":
        if not _ensure_module_available("transformers", "transformers>=4.34"):
            fallback_reason = "transformers dependency is missing"
            reason = f"{reason}; {fallback_reason}" if reason else fallback_reason
            requested = "bilstm"
        else:
            try:
                importlib.import_module("transformers")
            except Exception as exc:  # pragma: no cover - defensive guard
                message = str(exc).splitlines()[0]
                fallback_reason = f"failed to import transformers ({message})"
                reason = f"{reason}; {fallback_reason}" if reason else fallback_reason
                requested = "bilstm"
            else:  # pragma: no cover - optional vision dependency check
                try:
                    importlib.import_module("torchvision")
                except Exception as exc:
                    message = str(exc).splitlines()[0]
                    fallback_reason = f"torchvision backend unavailable ({message})"
                    reason = f"{reason}; {fallback_reason}" if reason else fallback_reason
                    requested = "bilstm"

    if reason is not None and reason.startswith(";"):
        reason = reason.lstrip("; ")
    return requested, reason


def read_dataset(path: Path) -> tuple[list[str], list[str], list[dict[str, str]]]:
    texts: list[str] = []
    labels: list[str] = []
    metadata: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text_raw = row.get("text")
            label_raw = row.get("label")
            if text_raw is None or label_raw is None:
                # Skip malformed rows that are missing required columns.
                continue
            text = str(text_raw).strip()
            label = str(label_raw).strip()
            if not text or not label:
                continue
            texts.append(text)
            labels.append(label)
            record: dict[str, str] = {}
            for key, value in row.items():
                if key in {"text", "label"}:
                    continue
                if key is None:
                    continue
                key_str = str(key).strip()
                if not key_str:
                    continue
                if value is None:
                    cleaned = ""
                else:
                    cleaned = str(value).strip()
                record[key_str] = cleaned
            metadata.append(record)
    return texts, labels, metadata


def read_unlabeled_dataset(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "text" not in fieldnames:
            raise ValueError(f"Unlabeled dataset at {path} must contain a 'text' column.")
        for row in reader:
            raw = row.get("text", "")
            text = str(raw).strip()
            if text:
                texts.append(text)
    return texts
def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "'": "'",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": "\"",
        "\u201d": "\"",
        "\u2013": "-",
        "\u2014": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.lower()


def scale_confidence_weight(
    base_weight: float,
    confidence: float,
    min_confidence: float,
    power: float,
    max_multiplier: float = float("inf"),
) -> float:
    # Scale a sample weight based on confidence while capping explosive growth.

    if base_weight <= 0:
        return 0.0
    safe_min = max(min_confidence, 1e-6)
    safe_conf = min(max(confidence, safe_min), 1.0)
    safe_power = max(power, 0.0)
    ratio = safe_conf / safe_min
    scaled = base_weight * (ratio ** safe_power)
    if math.isfinite(max_multiplier) and max_multiplier > 0:
        ceiling = base_weight * max_multiplier
        return min(scaled, ceiling)
    return scaled


def apply_auto_optimizations(
    args: argparse.Namespace,
    *,
    dataset_size: int,
    num_labels: int,
    using_cuda: bool,
    using_mps: bool,
    amp_available: bool,
    memory_guard: bool = False,
    performance_overdrive: bool = False,
) -> list[str]:
    actions: list[str] = []
    if not getattr(args, "auto_optimizations", True):
        return actions

    if using_cuda or using_mps:
        if amp_available and not args.fp16:
            args.fp16 = True
            actions.append("enable_fp16")
            device_label = "CUDA" if using_cuda else "MPS"
            print(f"Auto-optimizations: enabled mixed precision (fp16) for {device_label} training.")

        free_gb = _safe_float(getattr(args, "cuda_memory_free_gb", None))
        dataset_cap = dataset_size if dataset_size > 0 else None
        current_batch = max(1, int(args.batch_size))
        min_batch = 256
        max_batch = 512
        if dataset_cap is not None:
            max_batch = min(max_batch, dataset_cap)
            if dataset_cap < min_batch:
                min_batch = dataset_cap
        target_batch = current_batch
        if target_batch > max_batch:
            target_batch = max_batch
        if target_batch < min_batch and (dataset_cap is None or dataset_cap >= min_batch):
            target_batch = min_batch
        if free_gb is not None:
            if free_gb < 2.0:
                target_batch = min(target_batch, max(128, min_batch, 256))
            elif free_gb < 4.0:
                target_batch = min(target_batch, max(min_batch, 384))
        if memory_guard and target_batch > current_batch:
            target_batch = current_batch
        if target_batch != current_batch:
            headroom_reason = "VRAM headroom" if (free_gb is not None and free_gb < 4.0) else "stable throughput"
            free_display = f"{free_gb:.2f} GiB" if free_gb is not None else "unknown"
            args.batch_size = target_batch
            actions.append(f"batch_size->{target_batch}")
            print(
                "Auto-optimizations: clamped batch size to {target} ({reason}; free VRAM ≈ {free}).".format(
                    target=target_batch,
                    reason=headroom_reason,
                    free=free_display,
                )
            )
            final_batch = int(getattr(args, "final_train_batch_size", 0) or 0)
            if final_batch <= 0 or final_batch > target_batch:
                args.final_train_batch_size = target_batch
                actions.append(f"final_batch->{target_batch}")

        grad_target = max(4, int(args.grad_accumulation_steps))
        if dataset_size > 30000:
            grad_target = max(8, grad_target)
        if args.grad_accumulation_steps != grad_target:
            args.grad_accumulation_steps = grad_target
            actions.append(f"grad_accumulation->{grad_target}")
            print(
                "Auto-optimizations: using {steps} gradient accumulation steps to preserve effective batch size.".format(
                    steps=grad_target
                )
            )

        if args.ema_decay > 0:
            args.ema_decay = 0.0
            args.ema_use_for_eval = False
            actions.append("ema->disabled")
            print("Auto-optimizations: disabled EMA to keep stochastic weight averaging as the sole averaging method.")

        if args.epochs >= 5:
            if args.swa_start_epoch <= 0 or args.swa_start_epoch >= args.epochs:
                swa_window = max(1, int(round(args.epochs * 0.15)))
                args.swa_start_epoch = max(args.epochs - swa_window, 1)
                actions.append(f"swa_start->{args.swa_start_epoch}")
            if args.swa_lr <= 0:
                args.swa_lr = max(args.learning_rate * 0.3, 5e-6)
                actions.append(f"swa_lr->{args.swa_lr:.2e}")
            print(
                "Auto-optimizations: configuring SWA for the final {window} epoch(s) with lr {lr:.2e}.".format(
                    window=max(1, args.epochs - args.swa_start_epoch + 1),
                    lr=args.swa_lr,
                )
            )
    else:
        if args.fp16 and amp_available:
            actions.append("fp16_disabled_cpu")
            args.fp16 = False
            print("Auto-optimizations: disabled fp16 because GPU/MPS acceleration is unavailable.")

        # On CPU the default batch size can be counter-productive.
        # Down-shift it automatically so that forward/backward passes complete faster.
        cpu_default_batch = DEFAULT_BATCH_SIZE
        current_batch = int(args.batch_size)
        if current_batch >= cpu_default_batch:
            if dataset_size >= 5000:
                target_batch = 32
            elif dataset_size >= 1000:
                target_batch = 24
            else:
                target_batch = 16
            target_batch = max(16, min(target_batch, dataset_size))
            if target_batch < current_batch:
                args.batch_size = target_batch
                actions.append(f"cpu_batch_size->{target_batch}")
                print(
                    "Auto-optimizations: lowered CPU batch size to "
                    f"{target_batch} to keep iteration times manageable."
                )

        # When no explicit worker pool is requested, spin up a small CPU loader pool.
        worker_spec = getattr(args, "dataloader_workers", None)
        try:
            worker_spec_int = None if worker_spec is None else int(worker_spec)
        except (TypeError, ValueError):
            worker_spec_int = None
        if not memory_guard:
            if worker_spec_int is None or worker_spec_int <= 0:
                suggested_workers = _auto_dataloader_workers(
                    performance_overdrive=performance_overdrive
                )
                if os.name == "nt" and not performance_overdrive:
                    suggested_workers = min(suggested_workers, 2)
                elif not performance_overdrive:
                    suggested_workers = min(suggested_workers, 4)
                if dataset_size < 1024 and not performance_overdrive:
                    suggested_workers = min(suggested_workers, 1)
                if suggested_workers > 0:
                    args.dataloader_workers = suggested_workers
                    actions.append(f"cpu_workers->{suggested_workers}")
                    print(
                        "Auto-optimizations: enabled "
                        f"{suggested_workers} CPU data loader worker(s) for background prefetching."
                    )

            # Allow modest buffering when background workers are active.
            if getattr(args, "dataloader_workers", 0):
                if getattr(args, "dataloader_prefetch", None) is not None:
                    args.dataloader_prefetch = max(1, min(2, int(args.dataloader_prefetch)))

    return actions


def _resolve_micro_batch_size(
    args: argparse.Namespace,
    base_batch: int,
    *,
    using_cuda: bool,
    using_mps: bool,
) -> int:
    """Determine an initial micro-batch size based on device headroom."""

    if base_batch <= 0:
        base_batch = 1
    override = getattr(args, "micro_batch_size", 0) or 0
    if override < 0:
        override = 0
    if override:
        return max(1, min(base_batch, int(override)))
    if not (using_cuda or using_mps):
        return base_batch

    free_gb = _safe_float(getattr(args, "cuda_memory_free_gb", None))
    total_gb = _safe_float(getattr(args, "cuda_memory_total_gb", None))
    if free_gb is None or free_gb <= 0:
        if using_cuda:
            return max(1, min(base_batch, max(32, base_batch // 2)))
        return base_batch

    utilisation = None
    if total_gb and total_gb > 0:
        utilisation = (total_gb - free_gb) / max(total_gb, 1e-6)

    candidate = base_batch
    if free_gb < 1.2:
        candidate = max(8, min(candidate, 32))
    elif free_gb < 2.0:
        candidate = max(16, min(candidate, 48))
    elif free_gb < 3.0:
        candidate = max(24, min(candidate, 64))
    elif free_gb < 4.5:
        candidate = max(32, min(candidate, 96))
    elif utilisation is not None:
        if utilisation > 0.93:
            candidate = max(48, min(candidate, max(1, base_batch // 3)))
        elif utilisation > 0.87:
            candidate = max(64, min(candidate, max(1, base_batch // 2)))

    return max(1, min(base_batch, candidate))


def tokenize(text: str) -> list[str]:
    normalised = normalise_text(text)
    return TOKEN_PATTERN.findall(normalised)


BIGRAM_TOKEN_PREFIX = "§bg:"
TRIGRAM_TOKEN_PREFIX = "§tg:"
CHAR_NGRAM_TOKEN_PREFIX = "§ch:"


@dataclass(frozen=True)
class VocabularyConfig:
    include_bigrams: bool = True
    include_trigrams: bool = False
    include_char_ngrams: bool = True
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    char_ngram_limit: int = 3


def _augment_tokens(base_tokens: Sequence[str], config: VocabularyConfig) -> list[str]:
    augmented: list[str] = list(base_tokens)
    token_count = len(base_tokens)
    if config.include_bigrams and token_count >= 2:
        augmented.extend(
            f"{BIGRAM_TOKEN_PREFIX}{first}_{second}"
            for first, second in zip(base_tokens, base_tokens[1:])
        )
    if config.include_trigrams and token_count >= 3:
        augmented.extend(
            f"{TRIGRAM_TOKEN_PREFIX}{first}_{second}_{third}"
            for first, second, third in zip(base_tokens, base_tokens[1:], base_tokens[2:])
        )
    if (
        config.include_char_ngrams
        and config.char_ngram_max >= config.char_ngram_min >= 1
    ):
        limit = max(0, int(config.char_ngram_limit))
        for token in base_tokens:
            padded = f"^{token}$"
            char_tokens: list[str] = []
            seen: set[str] = set()
            length = len(padded)
            for size in range(config.char_ngram_min, config.char_ngram_max + 1):
                if size > length:
                    break
                for idx in range(length - size + 1):
                    gram = padded[idx : idx + size]
                    marker = f"{CHAR_NGRAM_TOKEN_PREFIX}{size}:{gram}"
                    if marker in seen:
                        continue
                    seen.add(marker)
                    char_tokens.append(marker)
            if limit > 0:
                char_tokens = char_tokens[:limit]
            augmented.extend(char_tokens)
    return augmented


def generate_training_tokens(text: str, config: VocabularyConfig) -> list[str]:
    base_tokens = tokenize(text)
    if not base_tokens:
        return []
    if not (
        config.include_bigrams
        or config.include_trigrams
        or config.include_char_ngrams
    ):
        return list(base_tokens)
    return _augment_tokens(base_tokens, config)


def build_vocab(
    texts: Sequence[str],
    min_freq: int = 1,
    config: VocabularyConfig | None = None,
    *,
    extra_texts: Sequence[str] | None = None,
    max_size: int | None = None,
) -> dict[str, int]:
    vocab_config = config or VocabularyConfig()
    counter: Counter[str] = Counter()
    corpus_iterable: list[str] = list(texts)
    if extra_texts:
        for candidate in extra_texts:
            if candidate:
                corpus_iterable.append(candidate)
    for text in corpus_iterable:
        counter.update(generate_training_tokens(text, vocab_config))

    vocab: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    limit = None
    if max_size is not None and max_size > 0:
        limit = max(max_size, len(vocab))
    for token, freq in counter.most_common():
        if freq < min_freq or token in vocab:
            continue
        if limit is not None and len(vocab) >= limit:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(
    text: str,
    vocab: dict[str, int],
    max_len: int,
    config: VocabularyConfig | None = None,
) -> list[int]:
    tokens = generate_training_tokens(text, config or VocabularyConfig())
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens[:max_len]]
    if len(token_ids) < max_len:
        token_ids.extend([vocab[PAD_TOKEN]] * (max_len - len(token_ids)))
    return token_ids


def stratified_split(indices: Sequence[int], labels: Sequence[str], *,
                     test_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in zip(indices, labels):
        by_label[label].append(idx)

    rng = random.Random(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label_indices in by_label.values():
        rng.shuffle(label_indices)
        split = max(1, int(round(len(label_indices) * (1 - test_ratio))))
        train_indices.extend(label_indices[:split])
        test_indices.extend(label_indices[split:])
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def stratified_kfold(indices: Sequence[int], labels: Sequence[str], *,
                     n_splits: int, seed: int) -> list[tuple[list[int], list[int]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for k-fold cross-validation.")
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in zip(indices, labels):
        by_label[label].append(idx)

    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for label_indices in by_label.values():
        rng.shuffle(label_indices)
        for position, example_idx in enumerate(label_indices):
            folds[position % n_splits].append(example_idx)

    splits: list[tuple[list[int], list[int]]] = []
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


def compute_pseudo_weight(
    base_weight: float,
    confidence: float,
    threshold: float,
    power: float,
    max_multiplier: float,
    *,
    consistency: float | None = None,
    consistency_floor: float = 0.0,
    consistency_power: float = 1.0,
) -> float:
    safe_threshold = max(threshold, 1e-6)
    ratio = max(confidence / safe_threshold, 1.0)
    weight = base_weight * (ratio ** power if power != 0 else 1.0)
    if consistency is not None:
        safe_floor = max(0.0, min(consistency_floor, 1.0))
        bounded = max(safe_floor, min(consistency, 1.0))
        exponent = max(0.0, consistency_power)
        if exponent != 1.0:
            bounded = bounded ** exponent
        weight = weight * bounded
    return min(weight, base_weight * max_multiplier)


def compute_consistency_score(agreement: float, std: float, max_std: float) -> float:
    agreement = max(0.0, min(1.0, agreement))
    if not math.isfinite(max_std) or max_std <= 0:
        return agreement
    scale = max(max_std, 1e-6)
    normalized_std = min(max(std / scale, 0.0), 1.0)
    stability = 1.0 - 0.5 * normalized_std
    return max(0.0, min(1.0, agreement * stability))


@dataclass
class LabelNgramModel:
    label: str
    order: int
    transitions: dict[tuple[str, ...], tuple[list[str], list[float]]]
    start_states: tuple[list[tuple[str, ...]], list[float]]


@dataclass
class SelfPlayCandidateEvaluation:
    label: str
    blended_confidence: float
    deterministic_confidence: float
    mc_confidence: float
    consistency: float
    margin: float
    top_predictions: list[tuple[str, float]]
    average_distribution: dict[str, Any]


@dataclass
class PseudoLabelDecision:
    text: str
    label: str
    confidence: float
    consistency: float
    agreement: float
    std: float


def build_label_ngram_models(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    order: int = 3,
) -> dict[str, LabelNgramModel]:
    order = max(1, order)
    sequences: dict[str, list[list[str]]] = defaultdict(list)
    for text, label in zip(texts, labels):
        tokens = tokenize(text)
        if tokens:
            sequences[label].append(tokens)

    models: dict[str, LabelNgramModel] = {}
    for label, label_sequences in sequences.items():
        transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        start_counter: Counter[tuple[str, ...]] = Counter()
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
        prepared: dict[tuple[str, ...], tuple[list[str], list[float]]] = {}
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
) -> list[str]:
    if not model.transitions:
        return []
    safe_max = max(1, max_tokens)
    order = model.order
    starts, weights = model.start_states
    if starts:
        prefix = tuple(rng.choices(starts, weights=weights, k=1)[0])
    else:
        prefix = tuple([BOS_TOKEN] * (order - 1)) if order > 1 else tuple()
    generated: list[str] = []
    inv_temp: float | None
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


def _orion_seed_rng(seed: str) -> random.Random:
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    seed_int = int(digest[:12], 16)
    return random.Random(seed_int)


def inspect_text_characteristics(text: str) -> dict[str, Any]:
    cleaned = " ".join(text.strip().split())
    lowered = cleaned.lower()
    tokens = re.findall(r"[A-Za-z0-9']+", cleaned)
    lower_tokens = [token.lower() for token in tokens]
    punctuation_counts: Counter[str] = Counter(ch for ch in cleaned if ch in string.punctuation)
    comma_count = punctuation_counts.get(",", 0)
    question_marks = punctuation_counts.get("?", 0)
    exclamation_marks = punctuation_counts.get("!", 0)
    uppercase_tokens = [token for token in tokens if len(token) > 1 and token.isupper()]
    capitalised_tokens = [
        token for token in tokens if token[:1].isupper() and not token.isupper()
    ]
    letters = [ch for ch in cleaned if ch.isalpha()]
    uppercase_ratio = (
        sum(1 for ch in letters if ch.isupper()) / len(letters) if letters else 0.0
    )
    question_triggers = {
        "what",
        "why",
        "how",
        "who",
        "where",
        "when",
        "which",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "will",
        "are",
        "is",
        "am",
        "should",
        "have",
        "has",
        "had",
    }
    leading_question_word = lower_tokens[0] in question_triggers if lower_tokens else False
    nearby_question_word = any(
        token in question_triggers for token in lower_tokens[: min(len(lower_tokens), 4)]
    )
    trailing_question_cue = bool(
        re.search(r"(?:right|won't you|don't you think|isn't it|okay)\s*\??$", lowered)
    )
    question_like_without_punctuation = nearby_question_word and question_marks == 0
    likely_question = question_marks > 0 or leading_question_word or trailing_question_cue
    false_question = (
        question_like_without_punctuation
        or (question_marks > 0 and not nearby_question_word and not leading_question_word)
        or trailing_question_cue
    )
    seduction_keywords = {
        "darling",
        "dear",
        "honey",
        "sweet",
        "sweetheart",
        "gorgeous",
        "handsome",
        "lovely",
        "beautiful",
        "cutie",
        "babe",
        "baby",
        "sugar",
        "tempt",
        "seduce",
        "allure",
        "charm",
        "flirt",
        "precious",
    }
    seduction_terms = [token for token in lower_tokens if token in seduction_keywords]
    seduction_score = len(seduction_terms)
    false_seduction = bool(seduction_terms) and (false_question or not likely_question)
    seduction_style = "none"
    if seduction_score:
        if false_seduction:
            seduction_style = "coaxing"
        elif seduction_score > 2:
            seduction_style = "intense"
        else:
            seduction_style = "warm"
    if likely_question:
        if false_question:
            question_type = "rhetorical"
        elif question_marks > 1 or "?!?" in cleaned or cleaned.count("?!") > 0:
            question_type = "emotional"
        else:
            question_type = "direct"
    else:
        question_type = "statement"
    return {
        "cleaned": cleaned,
        "lowered": lowered,
        "tokens": tokens,
        "lower_tokens": lower_tokens,
        "comma_count": comma_count,
        "punctuation_counts": dict(punctuation_counts),
        "question_marks": question_marks,
        "exclamation_marks": exclamation_marks,
        "uppercase_tokens": uppercase_tokens,
        "capitalised_tokens": capitalised_tokens,
        "uppercase_ratio": uppercase_ratio,
        "likely_question": likely_question,
        "false_question": false_question,
        "question_type": question_type,
        "seduction_terms": seduction_terms,
        "seduction_score": seduction_score,
        "false_seduction": false_seduction,
        "seduction_style": seduction_style,
        "trailing_question_cue": trailing_question_cue,
        "question_like_without_punctuation": question_like_without_punctuation,
    }


def craft_orion_reflections(
    features: Mapping[str, Any],
    *,
    label: str | None,
    rng: random.Random,
    context: str,
) -> list[str]:
    reflections: list[str] = []
    comma_count = int(features.get("comma_count", 0))
    if comma_count <= 0:
        reflections.append("Orion notices the breath racing without commas and wonders where to pause.")
    else:
        reflections.append(
            f"Orion counts {comma_count} comma{'s' if comma_count != 1 else ''} and feels the pauses like stepping stones."
        )
    uppercase_tokens = list(features.get("uppercase_tokens", []))
    if uppercase_tokens:
        highlighted = ", ".join(uppercase_tokens[:3])
        reflections.append(f"Capital letters {highlighted} flare like constellations—why do they shout here?")
    else:
        reflections.append("Lowercase words murmur throughout, so Orion leans in to catch the whisper.")
    question_type = str(features.get("question_type", "statement"))
    if question_type == "direct":
        reflections.append("It arrives as a direct question, inviting Orion to answer honestly.")
    elif question_type == "emotional":
        reflections.append("The punctuation ripples with emotion—Orion steadies the signal before replying.")
    elif question_type == "rhetorical":
        reflections.append("It dresses like a question yet feels declarative; Orion checks if a real answer is wanted.")
    else:
        reflections.append("No question pulses here, just a statement to absorb fully.")
    seduction_terms = list(features.get("seduction_terms", []))
    if seduction_terms:
        preview = ", ".join(sorted(set(seduction_terms))[:3])
        if bool(features.get("false_seduction")):
            reflections.append(
                f"Soft lures ({preview}) curl around the syntax, but Orion tests whether they mask true intent."
            )
        else:
            reflections.append(
                f"Warm phrases ({preview}) color the tone; Orion balances affection with clarity."
            )
    else:
        reflections.append("No seductive haze detected—only the raw contour of meaning.")
    uppercase_ratio = float(features.get("uppercase_ratio", 0.0))
    if uppercase_ratio > 0.4:
        reflections.append("Uppercase intensity spikes, so Orion softens the response to bring balance.")
    elif uppercase_ratio > 0.15:
        reflections.append("Capital emphasis flickers; Orion matches the energy without overpowering it.")
    else:
        reflections.append("The tone stays gentle, giving Orion space to explore details.")
    if context == "self_play" and label:
        humanised = label.replace("_", " ")
        reflections.append(f"Orion tags this exploration under {humanised} and records the sensations.")
    rng.shuffle(reflections)
    return reflections


def enrich_with_orion_exploration(text: str, label: str, rng: random.Random) -> str:
    features = inspect_text_characteristics(text)
    reflections = craft_orion_reflections(features, label=label, rng=rng, context="self_play")
    selected = reflections[:3]
    reflection_blurb = " ".join(selected)
    if reflection_blurb:
        return f"{text} Orion reflects: {reflection_blurb}"
    return text


def build_advanced_valuation_suite() -> list[str]:
    canonical_cases = [
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
    introspective_cases = [
        "Orion pauses mid-sentence asking, what is a comma really doing here—is it a breath, a hinge, or a cage for meaning?",
        "This update shouts in CAPITAL LETTERS and barely whispers a question; help me decode whether it's urgency or just noise.",
        "I'm not really asking, you're going to fix this for me gorgeous, you will help, right.",
        "Explain how to spot a false question that seduces agreement versus a real plea for help in this conversation.",
        "Why do the headlines scream in uppercase while the body hides the point—guide Orion through the big-letter labyrinth.",
        "Someone keeps flattering me without asking anything real; show me how to separate false seduction from genuine questions.",
    ]
    return canonical_cases + introspective_cases


def build_label_concept_library(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    label_to_idx: dict[str, int],
    max_keywords: int = 16,
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    """Construct per-label lexical concept profiles using TF-IDF weights."""

    num_classes = len(label_to_idx)
    lexical_profiles: list[dict[str, Any]] = [dict() for _ in range(num_classes)]
    keywords: list[list[str]] = [[] for _ in range(num_classes)]
    token_counts: dict[str, Counter[str]] = {label: Counter() for label in label_to_idx}
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
        weighted: dict[str, Any] = {}
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
    vocab: dict[str, int],
    label_to_idx: dict[str, int],
    max_len: int,
    device: TorchDevice,
    tokenizer: Any | None = None,
    tokenizer_cache: Callable[[str], tuple[Sequence[int], Sequence[int]]] | None = None,
    embedding_model: Callable[[str], VectorLike] | None = None,
    samples: int = 4,
    vocab_config: VocabularyConfig | None = None,
    emotion_encoder: EmotionLexicon | None = None,
    emotion_dim: int = 0,
    emotion_config: EmotionTrainingConfig | None = None,
    metadata_encoder: StructuredMetadataEncoder | None = None,
    lexicon_dim: int = 0,
    metadata_dim: int = 0,
    keyword_calibrator: KeywordIntentCalibrator | None = None,
    symbolic_router: CognitiveIntentRouter | None = None,
    meta_stacker: MetaIntentStacker | None = None,
) -> SelfPlayCandidateEvaluation | None:
    shots = max(1, samples)
    ids, mask, emotion_features = _prepare_model_inputs(
        text,
        vocab=vocab,
        max_len=max_len,
        device=device,
        tokenizer=tokenizer,
        tokenizer_cache=tokenizer_cache,
        embedding_model=embedding_model,
        vocab_config=vocab_config,
        emotion_encoder=emotion_encoder,
        emotion_dim=emotion_dim,
        metadata=None,
        metadata_encoder=metadata_encoder,
        lexicon_dim=lexicon_dim,
        metadata_dim=metadata_dim,
    )
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    counts: Counter[str] = Counter()
    confidences: list[float] = []
    margins: list[float] = []
    prob_sums: dict[str, Any] = defaultdict(float)
    was_training = model.training
    model.train()
    adjustment_vector = compose_logit_adjustments(
        text,
        calibrator=keyword_calibrator,
        router=symbolic_router,
    )
    keyword_adjustment_tensor: TorchTensor | None = None
    if adjustment_vector:
        keyword_adjustment_tensor = torch.tensor(
            adjustment_vector,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
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

                base_logits = logits
                if keyword_adjustment_tensor is not None:
                    logits = logits + keyword_adjustment_tensor.to(
                        dtype=logits.dtype, device=logits.device
                    )
                if meta_stacker is not None:
                    meta_adjustment = meta_stacker.compute_adjustment(
                        base_logits,
                        keyword_adjustment_tensor[0]
                        if keyword_adjustment_tensor is not None
                        else None,
                    )
                    if meta_adjustment:
                        meta_tensor = torch.tensor(
                            meta_adjustment,
                            dtype=logits.dtype,
                            device=logits.device,
                        ).unsqueeze(0)
                        logits = logits + meta_tensor

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
        vocab_config=vocab_config,
        emotion_encoder=emotion_encoder,
        emotion_dim=emotion_dim,
        emotion_config=emotion_config,
        metadata=None,
        metadata_encoder=metadata_encoder,
        lexicon_dim=lexicon_dim,
        metadata_dim=metadata_dim,
        calibrator=keyword_calibrator,
        symbolic_router=symbolic_router,
        meta_stacker=meta_stacker,
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
    student_logits: TorchTensor,
    teacher_logits: TorchTensor,
    *,
    temperature: float,
) -> TorchTensor:
    safe_temperature = max(temperature, 1e-4)
    student_log_probs = F.log_softmax(student_logits / safe_temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / safe_temperature, dim=-1)
    kd = F.kl_div(student_log_probs, teacher_probs, reduction="none")
    if kd.dim() == 2:
        kd = kd.sum(dim=1)
    return kd * (safe_temperature ** 2)


def symmetric_kl_divergence(
    logits_a: TorchTensor,
    logits_b: TorchTensor,
) -> TorchTensor:
    """Symmetric KL divergence between two logits tensors (per example)."""

    log_prob_a = F.log_softmax(logits_a, dim=-1)
    log_prob_b = F.log_softmax(logits_b, dim=-1)
    prob_a = log_prob_a.exp()
    prob_b = log_prob_b.exp()
    kl_ab = F.kl_div(log_prob_a, prob_b, reduction="none").sum(dim=-1)
    kl_ba = F.kl_div(log_prob_b, prob_a, reduction="none").sum(dim=-1)
    return 0.5 * (kl_ab + kl_ba)


class CpuOffloadedEmaModel(nn.Module):
    """EMA weights stored on CPU to minimise GPU footprint."""

    def __init__(
        self,
        model: nn.Module,
        decay: float,
        *,
        dtype: TorchDType = torch.float32,
    ) -> None:
        super().__init__()
        self.decay = float(max(0.0, min(decay, 0.99999)))
        self.offload_dtype = dtype
        self.module = copy.deepcopy(model)
        self.n_averaged: int = 0
        self._offload()
        self.train(model.training)

    def forward(self, *args: Any, **kwargs: Any):
        return self.module(*args, **kwargs)

    def _infer_device(self) -> TorchDevice:
        for param in self.module.parameters():
            return param.device
        for buffer in self.module.buffers():
            return buffer.device
        return torch.device("cpu")

    def _offload(self) -> None:
        self.module.to(torch.device("cpu"))
        for param in self.module.parameters():
            if param.is_floating_point():
                param.data = param.data.to(dtype=self.offload_dtype)
            else:
                param.data = param.data.cpu()
        for buffer in self.module.buffers():
            buffer.data = buffer.data.cpu()

    @torch.no_grad()
    def update_parameters(self, model: nn.Module) -> None:
        source_params = dict(model.named_parameters())
        source_buffers = dict(model.named_buffers())
        if self.n_averaged == 0:
            for name, avg_param in self.module.named_parameters():
                source = source_params.get(name)
                if source is None:
                    continue
                data = source.detach()
                if data.is_floating_point():
                    avg_param.copy_(data.to(dtype=self.offload_dtype, device=torch.device("cpu")))
                else:
                    avg_param.copy_(data.cpu())
            for name, avg_buffer in self.module.named_buffers():
                source_buffer = source_buffers.get(name)
                if source_buffer is not None:
                    avg_buffer.copy_(source_buffer.detach().cpu())
            self.n_averaged = 1
            return
        decay = self.decay
        one_minus_decay = 1.0 - decay
        for name, avg_param in self.module.named_parameters():
            source = source_params.get(name)
            if source is None:
                continue
            data = source.detach()
            if data.is_floating_point():
                cpu_value = data.to(dtype=self.offload_dtype, device=torch.device("cpu"))
                avg_param.mul_(decay).add_(cpu_value, alpha=one_minus_decay)
            else:
                avg_param.copy_(data.cpu())
        for name, avg_buffer in self.module.named_buffers():
            source_buffer = source_buffers.get(name)
            if source_buffer is not None:
                avg_buffer.copy_(source_buffer.detach().cpu())
        self.n_averaged += 1

    def to(self, *args: Any, **kwargs: Any):
        result = super().to(*args, **kwargs)
        target_device = kwargs.get("device")
        if target_device is None:
            for arg in args:
                if isinstance(arg, TorchDevice):
                    target_device = arg
                    break
                if isinstance(arg, str):
                    target_device = torch.device(arg)
                    break
        if target_device is not None and target_device.type == "cpu":
            self._offload()
        return result

    def state_dict(self, *args: Any, **kwargs: Any):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any):
        return self.module.load_state_dict(*args, **kwargs)

    def train(self, mode: bool = True):
        self.module.train(mode)
        return super().train(mode)

    def eval(self):
        self.module.eval()
        return super().eval()

    @contextlib.contextmanager
    def activate(self, device: TorchDevice):
        previous_device = self._infer_device()
        needs_transfer = previous_device.type == "cpu" and device.type != "cpu"
        if needs_transfer:
            self.module.to(device)
        try:
            yield self
        finally:
            if needs_transfer:
                self._offload()
                if device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass


def create_ema_model(
    model: nn.Module,
    decay: float,
    *,
    offload_to_cpu: bool = False,
    offload_dtype: TorchDType = torch.float32,
) -> nn.Module:
    def ema_average(param_avg: TorchTensor, param: TorchTensor, num_averaged: int) -> TorchTensor:
        if num_averaged == 0:
            return param.detach()
        return param_avg * decay + param.detach() * (1.0 - decay)

    if offload_to_cpu:
        return CpuOffloadedEmaModel(model, decay, dtype=offload_dtype)

    ema_model = AveragedModel(model, avg_fn=ema_average)
    module_attr = getattr(ema_model, "module", None)
    if not isinstance(module_attr, nn.Module):
        raise RuntimeError("AveragedModel failed to expose a wrapped module instance.")
    module_to_update = module_attr
    module_to_update.load_state_dict(model.state_dict())
    return ema_model


def clone_model_state_dict(model: nn.Module) -> dict[str, Any]:
    # Detach and copy a model's parameters to CPU, unwrapping wrappers first.
    module = model
    seen: set[int] = set()
    while True:
        module_id = id(module)
        if module_id in seen:
            break
        seen.add(module_id)
        if isinstance(module, AveragedModel):
            inner = getattr(module, "module", None)
            if not isinstance(inner, nn.Module):
                raise RuntimeError("Encountered AveragedModel without a wrapped module.")
            module = inner
            continue
        child = getattr(module, "module", None)
        if isinstance(child, nn.Module) and child is not module:
            module = child
            continue
        break

    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _memory_efficient_optimizer(
    parameters: Iterable[nn.Parameter | dict[str, Any]],
    base_lr: float,
    weight_decay: float,
) -> TorchOptimizer:
    if _adafactor_available and adafactor_factory is not None:
        return adafactor_factory(
            parameters,
            lr=base_lr,
            weight_decay=weight_decay,
            relative_step=False,
            scale_parameter=False,
        )
    return torch.optim.AdamW(parameters, lr=base_lr, weight_decay=weight_decay)


def create_transformer_optimizer(
    model: nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
    layerwise_decay: float,
) -> TorchOptimizer:
    """Construct a memory-efficient optimizer with optional layer-wise learning-rate decay."""

    named_parameters = list(model.named_parameters())
    if not named_parameters:
        return _memory_efficient_optimizer(model.parameters(), base_lr, weight_decay)

    no_decay_tokens = (
        "bias",
        "LayerNorm.weight",
        "LayerNorm.bias",
        "layer_norm.weight",
        "layer_norm.bias",
        "layernorm.weight",
        "layernorm.bias",
        "norm.weight",
        "norm.bias",
        "bn.weight",
        "bn.bias",
    )

    def assign_layer_ids() -> tuple[dict[str, int], int]:
        patterns = [
            r"encoder\.layer\.(\d+)",
            r"transformer\.layer\.(\d+)",
            r"layers?\.(\d+)\.",
            r"block\.(\d+)\.",
        ]
        raw_ids: dict[str, int | None] = {}
        max_seen = 0
        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            assigned: int | None = None
            if any(token in name for token in ("embeddings", "embed_tokens", "wte", "wpe", "shared")):
                assigned = 0
            else:
                for pattern in patterns:
                    match = re.search(pattern, name)
                    if match:
                        assigned = int(match.group(1)) + 1
                        break
            raw_ids[name] = assigned
            if assigned is not None:
                max_seen = max(max_seen, assigned)
        default_id = max_seen + 1
        resolved_ids: dict[str, int] = {}
        for name, assigned in raw_ids.items():
            resolved = assigned if assigned is not None else default_id
            resolved_ids[name] = int(resolved)
        max_layer = max(resolved_ids.values()) if resolved_ids else 0
        return resolved_ids, max_layer

    use_layerwise = layerwise_decay > 0 and not math.isclose(layerwise_decay, 1.0, rel_tol=1e-6)
    if not use_layerwise:
        decay_params = [
            param
            for name, param in named_parameters
            if param.requires_grad and not any(token in name for token in no_decay_tokens)
        ]
        nodecay_params = [
            param
            for name, param in named_parameters
            if param.requires_grad and any(token in name for token in no_decay_tokens)
        ]
        param_groups: list[dict[str, Any]] = []
        if decay_params:
            param_groups.append({"params": decay_params, "lr": base_lr, "weight_decay": weight_decay})
        if nodecay_params:
            param_groups.append({"params": nodecay_params, "lr": base_lr, "weight_decay": 0.0})
        if not param_groups:
            param_groups.append({"params": [param for _, param in named_parameters], "lr": base_lr, "weight_decay": weight_decay})
        return _memory_efficient_optimizer(param_groups, base_lr, weight_decay)

    layer_ids, max_layer = assign_layer_ids()
    grouped: dict[tuple[float, float], list[nn.Parameter]] = defaultdict(list)
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        layer_id = layer_ids.get(name, max_layer)
        depth_delta = max_layer - layer_id
        scaled_lr = base_lr * (layerwise_decay ** depth_delta)
        decay_value = weight_decay
        if any(token in name for token in no_decay_tokens):
            decay_value = 0.0
        grouped[(float(scaled_lr), float(decay_value))].append(param)

    param_groups = [
        {"params": params, "lr": lr, "weight_decay": decay}
        for (lr, decay), params in grouped.items()
    ]
    return _memory_efficient_optimizer(param_groups, base_lr, weight_decay)


def apply_augmentation_strategy(tokens: list[str], strategy: str, rng: random.Random) -> list[str]:
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
    metadata: Sequence[dict[str, str] | None] | None = None,
) -> tuple[
    list[str],
    list[str],
    list[float],
    list[dict[str, str] | None] | None,
    int,
]:
    if probability <= 0 or max_copies <= 0 or not strategies:
        return (
            list(texts),
            list(labels),
            list(weights),
            list(metadata) if metadata is not None else None,
            0,
        )
    augmented_texts = list(texts)
    augmented_labels = list(labels)
    augmented_weights = list(weights)
    augmented_metadata = list(metadata) if metadata is not None else None
    augment_count = 0
    for idx, (text, label, weight) in enumerate(zip(texts, labels, weights)):
        base_tokens = tokenize(text)
        base_metadata = None
        if metadata is not None:
            if idx < len(metadata):
                base_metadata = metadata[idx]
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
            if augmented_metadata is not None:
                augmented_metadata.append(base_metadata)
            augment_count += 1
    return augmented_texts, augmented_labels, augmented_weights, augmented_metadata, augment_count


def compute_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    *,
    label_to_idx: dict[str, int],
) -> dict[str, Any]:
    total = len(targets)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    confusion: dict[str, dict[str, int]] = {
        idx_to_label[idx]: {idx_to_label[jdx]: 0 for jdx in idx_to_label}
        for idx in idx_to_label
    }
    per_label_counts: dict[str, dict[str, int]] = {
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

    per_label_metrics: dict[str, dict[str, Any]] = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    label_support: dict[str, int] = {idx_to_label[idx]: 0 for idx in idx_to_label}
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

    DEFAULT_LEXICON: dict[str, dict[str, Any]] = {
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

    EMOTIONS: tuple[str, ...] = (
        "joy",
        "trust",
        "fear",
        "surprise",
        "sadness",
        "disgust",
        "anger",
        "anticipation",
    )

    def __init__(self, lexicon: dict[str, dict[str, Any]] | None = None, *, smoothing: float = 1e-3) -> None:
        self.lexicon = lexicon or self.DEFAULT_LEXICON
        self.emotions: tuple[str, ...] = self.EMOTIONS
        self.emotion_to_idx: dict[str, int] = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        self.smoothing = float(max(smoothing, 1e-6))

    def vectorise(self, text: str) -> list[float]:
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

    def _compute_emphasis(self, text: str) -> dict[str, Any]:
        lowered = normalise_text(text)
        emphasis: dict[str, Any] = {}
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


@dataclass
class StructuredFeatureField:
    name: str
    offset: int
    value_to_index: dict[str, int]
    size: int
    missing_index: int | None
    values: list[str]


class StructuredMetadataEncoder:
    """Encode structured metadata columns into deterministic feature vectors."""

    def __init__(
        self,
        records: Sequence[Mapping[str, str]],
        *,
        min_frequency: int = 1,
        include_missing: bool = True,
    ) -> None:
        self.include_missing = bool(include_missing)
        self.fields: list[StructuredFeatureField] = []
        self.dimension: int = 0
        if not records:
            return
        frequency_threshold = max(1, int(min_frequency))
        field_value_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            if not record:
                continue
            for key, value in record.items():
                normalised = self._normalise_value(value)
                if normalised:
                    field_value_counts[key][normalised] += 1
                else:
                    field_value_counts[key]["__missing__"] += 1
        for field_name in sorted(field_value_counts):
            counter = field_value_counts[field_name]
            allowed: list[str] = [
                value
                for value, count in counter.items()
                if value != "__missing__" and count >= frequency_threshold
            ]
            if not allowed and not self.include_missing:
                continue
            allowed.sort()
            value_to_index: dict[str, int] = {value: idx for idx, value in enumerate(allowed)}
            missing_index: int | None = None
            size = len(allowed)
            if self.include_missing:
                missing_index = size
                size += 1
            field = StructuredFeatureField(
                name=field_name,
                offset=self.dimension,
                value_to_index=value_to_index,
                size=size,
                missing_index=missing_index,
                values=allowed,
            )
            self.fields.append(field)
            self.dimension += size

    @staticmethod
    def _normalise_value(value: str | None) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        return text

    def encode(self, record: Mapping[str, str] | None) -> list[float]:
        if not self.fields:
            return []
        vector = [0.0] * self.dimension
        for field in self.fields:
            slot = field.missing_index
            raw_value = None if record is None else record.get(field.name)
            normalised = self._normalise_value(raw_value)
            if normalised and normalised in field.value_to_index:
                slot = field.value_to_index[normalised]
            if slot is not None:
                vector[field.offset + slot] = 1.0
        return vector

    def export(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "fields": [
                {
                    "name": field.name,
                    "values": field.values,
                    "include_missing": self.include_missing,
                    "size": field.size,
                }
                for field in self.fields
            ],
        }


def compose_emotion_features(
    text: str,
    metadata: Mapping[str, str] | None,
    *,
    lexicon: EmotionLexicon | None,
    metadata_encoder: StructuredMetadataEncoder | None,
    lexicon_dim: int,
    metadata_dim: int,
) -> list[float]:
    components: list[float] = []
    if lexicon is not None and lexicon_dim > 0:
        lexicon_values = list(lexicon.vectorise(text))
        if len(lexicon_values) < lexicon_dim:
            lexicon_values.extend([0.0] * (lexicon_dim - len(lexicon_values)))
        elif len(lexicon_values) > lexicon_dim:
            lexicon_values = lexicon_values[:lexicon_dim]
        components.extend(lexicon_values)
    if metadata_encoder is not None and metadata_dim > 0:
        meta_values = metadata_encoder.encode(metadata)
        if len(meta_values) < metadata_dim:
            meta_values.extend([0.0] * (metadata_dim - len(meta_values)))
        elif len(meta_values) > metadata_dim:
            meta_values = meta_values[:metadata_dim]
        components.extend(meta_values)
    target_dim = max(0, lexicon_dim) + max(0, metadata_dim)
    if len(components) < target_dim:
        components.extend([0.0] * (target_dim - len(components)))
    return components


BIGRAM_SEPARATOR = "‖"


class KeywordIntentCalibrator:
    """Apply keyword-derived priors to stabilise intent predictions."""

    def __init__(
        self,
        *,
        label_to_idx: Mapping[str, int],
        bias: Sequence[float],
        feature_scores: Mapping[str, Sequence[float]],
        top_features: Mapping[int, Sequence[tuple[str, float]]],
        feature_weight: float,
        bias_weight: float,
        normalise_power: float,
        min_frequency: int,
        bigram_min_frequency: int,
        smoothing: float,
        strength_threshold: float,
    ) -> None:
        self.label_to_idx = dict(label_to_idx)
        self.idx_to_label = {
            idx: label for label, idx in self.label_to_idx.items()
        }
        self.num_classes = len(self.label_to_idx)
        self.bias: list[float] = list(bias)
        if len(self.bias) != self.num_classes:
            self.bias = [0.0] * self.num_classes
        self.feature_scores: dict[str, list[float]] = {
            feature: list(scores)
            for feature, scores in feature_scores.items()
        }
        self.feature_weight = float(feature_weight)
        self.bias_weight = float(bias_weight)
        self.normalise_power = float(max(normalise_power, 0.0))
        self.min_frequency = int(max(min_frequency, 1))
        self.bigram_min_frequency = int(max(bigram_min_frequency, 1))
        self.smoothing = float(max(smoothing, 1e-6))
        self.strength_threshold = float(max(strength_threshold, 0.0))
        self.top_features: dict[int, list[tuple[str, float]]] = {
            idx: list(entries)
            for idx, entries in top_features.items()
        }

    @property
    def feature_count(self) -> int:
        return len(self.feature_scores)

    def has_bias(self) -> bool:
        return any(abs(value) > 1e-6 for value in self.bias) and self.bias_weight != 0.0

    def _extract_features(self, tokens: Sequence[str]) -> set[str]:
        features: set[str] = set(tokens)
        if len(tokens) >= 2:
            for first, second in zip(tokens, tokens[1:]):
                if not first or not second:
                    continue
                features.add(f"{first}{BIGRAM_SEPARATOR}{second}")
        return features

    def vectorise(self, text: str) -> list[float]:
        tokens = tokenize(text)
        features = self._extract_features(tokens)
        contributions: list[float] | None = None
        matches = 0
        for feature in features:
            scores = self.feature_scores.get(feature)
            if scores is None:
                continue
            matches += 1
            if contributions is None:
                contributions = list(scores)
            else:
                contributions = [value + delta for value, delta in zip(contributions, scores)]
        adjusted: list[float] | None = None
        if contributions is not None and matches > 0:
            if self.normalise_power > 0:
                divisor = matches ** self.normalise_power
                if divisor <= 0:
                    divisor = 1.0
            else:
                divisor = 1.0
            scale = self.feature_weight / divisor if divisor != 0 else self.feature_weight
            adjusted = [value * scale for value in contributions]
        elif contributions is not None:
            adjusted = [value * self.feature_weight for value in contributions]
        bias_adjustment: list[float] | None = None
        if self.has_bias():
            bias_adjustment = [value * self.bias_weight for value in self.bias]
        if adjusted is None:
            if bias_adjustment is None:
                return []
            return list(bias_adjustment)
        if bias_adjustment is not None:
            return [value + bias for value, bias in zip(adjusted, bias_adjustment)]
        return adjusted

    def compute_adjustment(
        self,
        text: str,
        *,
        device: TorchDevice | None = None,
        dtype: TorchDType | None = None,
    ) -> TorchTensor | None:
        vector = self.vectorise(text)
        if not vector:
            return None
        return torch.tensor(vector, dtype=dtype or torch.float32, device=device)

    def export_metadata(self, top_k: int = 10) -> dict[str, Any]:
        feature_preview: dict[str, list[dict[str, Any]]] = {}
        for idx in range(self.num_classes):
            label = self.idx_to_label.get(idx, str(idx))
            entries: list[dict[str, Any]] = []
            for feature, score in self.top_features.get(idx, [])[:top_k]:
                entries.append({"feature": feature, "score": float(score)})
            feature_preview[label] = entries
        return {
            "enabled": True,
            "feature_count": self.feature_count,
            "bias_weight": self.bias_weight,
            "feature_weight": self.feature_weight,
            "normalise_power": self.normalise_power,
            "min_frequency": self.min_frequency,
            "bigram_min_frequency": self.bigram_min_frequency,
            "smoothing": self.smoothing,
            "strength_threshold": self.strength_threshold,
            "has_bias": self.has_bias(),
            "top_features": feature_preview,
        }


class CognitiveIntentRouter:
    """Blend hand-crafted neuro-symbolic cues into intent logits."""

    def __init__(
        self,
        *,
        label_to_idx: Mapping[str, int],
        signal_scale: float = 2.5,
        penalty_scale: float = 1.6,
        synergy_scale: float = 0.75,
    ) -> None:
        self.label_to_idx = dict(label_to_idx)
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        self.signal_scale = float(max(signal_scale, 0.0))
        self.penalty_scale = float(max(penalty_scale, 0.0))
        self.synergy_scale = float(max(synergy_scale, 0.0))

        self._request_indices = self._resolve_indices(["request"])
        self._resource_indices = self._resolve_indices(["resource_request", "coordination"])
        self._schedule_indices = self._resolve_indices(["schedule_update", "reminder"])
        self._gratitude_indices = self._resolve_indices(["thank_you", "morale_boost"])
        self._hazard_indices = self._resolve_indices(["hazard_alert"])
        self._feedback_indices = self._resolve_indices(["feedback"])
        self._check_in_indices = self._resolve_indices(["check_in"])
        self._invitation_indices = self._resolve_indices(["invitation"])
        self._question_indices = self._resolve_indices(["question"])
        self._reminder_indices = self._resolve_indices(["reminder"])
        self._progress_indices = self._resolve_indices(["progress_update", "schedule_update"])
        self._idea_indices = self._resolve_indices(["idea_proposal"])
        self._learning_indices = self._resolve_indices(["learning_share"])
        self._reflection_indices = self._resolve_indices(["reflection_prompt"])
        self._closure_indices = self._resolve_indices(["closure_summary"])
        self._apology_indices = self._resolve_indices(["apology"])
        self._greeting_indices = self._resolve_indices(["greeting", "farewell"])
        self._morale_indices = self._resolve_indices(["morale_boost"])

        self._question_tokens = {
            "what",
            "when",
            "where",
            "who",
            "why",
            "which",
            "how",
        }
        self._schedule_keywords = (
            "schedule",
            "time",
            "deadline",
            "calendar",
            "presenting",
            "tonight",
            "tomorrow",
            "reschedule",
            "meeting",
        )
        self._resource_keywords = (
            "upload",
            "share",
            "send",
            "submit",
            "archive",
            "assets",
            "folder",
            "drive",
            "location",
            "route",
            "delivery",
            "handoff",
            "credential",
            "password",
            "login",
            "database",
            "server",
            "staging",
            "reset",
            "access",
        )
        self._issue_keywords = (
            "issue",
            "problem",
            "broken",
            "crash",
            "error",
            "failing",
            "failure",
            "bug",
            "down",
            "outage",
            "incident",
        )
        self._gratitude_keywords = (
            "thank",
            "thanks",
            "appreciate",
            "grateful",
        )
        self._invitation_keywords = (
            "invite",
            "invited",
            "invitation",
            "join",
            "gather",
            "celebrate",
            "party",
        )
        self._reminder_keywords = (
            "remind",
            "reminder",
            "remember",
            "nudge",
        )
        self._progress_keywords = (
            "update",
            "progress",
            "status",
            "report",
            "briefing",
            "latest",
            "checklist",
        )
        self._hazard_keywords = (
            "hazard",
            "alert",
            "risk",
            "safety",
            "critical",
            "emergency",
        )
        self._idea_keywords = (
            "idea",
            "brainstorm",
            "proposal",
            "propose",
            "concept",
            "suggest",
        )
        self._learning_keywords = (
            "learned",
            "lesson",
            "insight",
            "share",
            "knowledge",
            "training",
        )
        self._reflection_keywords = (
            "reflect",
            "reflection",
            "retrospective",
            "ponder",
            "consider",
            "think back",
        )
        self._closure_keywords = (
            "summary",
            "summarize",
            "recap",
            "wrap up",
            "conclusion",
            "sign off",
            "close out",
        )
        self._apology_keywords = (
            "sorry",
            "apologize",
            "apologies",
            "regret",
        )
        self._morale_keywords = (
            "great work",
            "congrats",
            "celebrate",
            "proud",
            "kudos",
        )

        self.trigger_counts: Counter[str] = Counter()
        self.adjusted_examples = 0
        self.examples_with_positive = 0
        self.total_positive_triggers = 0
        self.total_negative_triggers = 0

    def reset_statistics(self) -> None:
        self.trigger_counts.clear()
        self.adjusted_examples = 0
        self.examples_with_positive = 0
        self.total_positive_triggers = 0
        self.total_negative_triggers = 0

    def _resolve_indices(self, labels: Sequence[str]) -> list[int]:
        indices: list[int] = []
        for label in labels:
            idx = self.label_to_idx.get(label)
            if idx is not None:
                indices.append(idx)
        return indices

    def _match_keywords(self, lowered: str, tokens: set[str], keywords: Sequence[str]) -> bool:
        for keyword in keywords:
            if " " in keyword:
                if keyword in lowered:
                    return True
                continue
            if keyword in tokens:
                return True
        return False

    def vectorise(self, text: str) -> list[float]:
        if self.num_classes == 0:
            return []
        lowered = unicodedata.normalize("NFKC", text).lower()
        tokens = set(re.findall(r"[a-z]+", lowered))
        contributions = [0.0] * self.num_classes
        triggered = False
        positive_triggers = 0

        def boost(indices: Sequence[int], weight: float, tag: str) -> None:
            nonlocal triggered, positive_triggers
            if not indices or weight <= 0:
                return
            triggered = True
            positive_triggers += 1
            synergy_bonus = self.synergy_scale * max(0, positive_triggers - 1)
            amount = weight + synergy_bonus
            for idx in indices:
                contributions[idx] += amount
            self.trigger_counts[tag] += 1
            self.total_positive_triggers += 1

        def penalise(indices: Sequence[int], weight: float, tag: str) -> None:
            nonlocal triggered
            if not indices or weight <= 0:
                return
            triggered = True
            for idx in indices:
                contributions[idx] -= weight
            self.trigger_counts[tag] += 1
            self.total_negative_triggers += 1

        is_question = "?" in text or bool(tokens & self._question_tokens)
        if is_question:
            boost(self._question_indices, self.signal_scale * 0.8, "question_form")
            boost(self._request_indices, self.signal_scale * 0.65, "question_request")
            penalise(self._check_in_indices, self.penalty_scale * 0.85, "question_vs_checkin")
            penalise(self._invitation_indices, self.penalty_scale * 0.7, "question_vs_invitation")
            penalise(self._greeting_indices, self.penalty_scale * 0.6, "question_vs_greeting")

        if self._match_keywords(lowered, tokens, self._schedule_keywords):
            boost(self._schedule_indices, self.signal_scale * 1.35, "schedule_lookup")
            boost(self._progress_indices, self.signal_scale * 0.75, "schedule_progress")

        if self._match_keywords(lowered, tokens, self._resource_keywords):
            boost(self._resource_indices, self.signal_scale * 1.25, "resource_lookup")
            boost(self._request_indices, self.signal_scale * 0.9, "resource_request")
            penalise(self._check_in_indices, self.penalty_scale * 0.5, "resource_vs_checkin")

        if self._match_keywords(lowered, tokens, self._issue_keywords):
            boost(self._hazard_indices, self.signal_scale * 1.3, "incident_signal")
            boost(self._feedback_indices, self.signal_scale * 1.1, "issue_feedback")
            penalise(self._invitation_indices, self.penalty_scale * 0.8, "incident_vs_invitation")
            penalise(self._morale_indices, self.penalty_scale * 0.7, "incident_vs_morale")

        if self._match_keywords(lowered, tokens, self._gratitude_keywords):
            boost(self._gratitude_indices, self.signal_scale * 1.45, "gratitude")
            penalise(self._hazard_indices, self.penalty_scale * 0.45, "gratitude_vs_alert")
            penalise(self._feedback_indices, self.penalty_scale * 0.45, "gratitude_vs_feedback")

        if self._match_keywords(lowered, tokens, self._invitation_keywords):
            boost(self._invitation_indices, self.signal_scale * 1.25, "invitation")
            penalise(self._hazard_indices, self.penalty_scale * 0.6, "invitation_vs_alert")

        if self._match_keywords(lowered, tokens, self._reminder_keywords):
            boost(self._reminder_indices, self.signal_scale * 1.2, "reminder")
            boost(self._schedule_indices, self.signal_scale * 0.85, "reminder_schedule")

        if self._match_keywords(lowered, tokens, self._progress_keywords):
            boost(self._progress_indices, self.signal_scale * 1.2, "progress_update")
            penalise(self._check_in_indices, self.penalty_scale * 0.55, "progress_vs_checkin")

        if self._match_keywords(lowered, tokens, self._hazard_keywords):
            boost(self._hazard_indices, self.signal_scale * 1.35, "hazard_alert")

        if self._match_keywords(lowered, tokens, self._idea_keywords):
            boost(self._idea_indices, self.signal_scale * 1.15, "idea_signal")

        if self._match_keywords(lowered, tokens, self._learning_keywords):
            boost(self._learning_indices, self.signal_scale, "learning_share")
            boost(self._reflection_indices, self.signal_scale * 0.85, "learning_reflection")

        if self._match_keywords(lowered, tokens, self._reflection_keywords):
            boost(self._reflection_indices, self.signal_scale * 1.05, "reflection")

        if self._match_keywords(lowered, tokens, self._closure_keywords):
            boost(self._closure_indices, self.signal_scale * 1.05, "closure_summary")

        if self._match_keywords(lowered, tokens, self._apology_keywords):
            boost(self._apology_indices, self.signal_scale * 1.25, "apology")
            penalise(self._hazard_indices, self.penalty_scale * 0.4, "apology_vs_alert")

        if self._match_keywords(lowered, tokens, self._morale_keywords):
            boost(self._gratitude_indices, self.signal_scale, "morale_support")

        if not triggered:
            return []
        self.adjusted_examples += 1
        if positive_triggers > 0:
            self.examples_with_positive += 1
        return contributions

    def export_metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "signal_scale": self.signal_scale,
            "penalty_scale": self.penalty_scale,
            "synergy_scale": self.synergy_scale,
            "examples_adjusted": int(self.adjusted_examples),
            "examples_with_positive": int(self.examples_with_positive),
            "positive_trigger_events": int(self.total_positive_triggers),
            "negative_trigger_events": int(self.total_negative_triggers),
            "trigger_counts": dict(sorted(self.trigger_counts.items())),
        }

    @property
    def total_triggers(self) -> int:
        return int(sum(self.trigger_counts.values()))


def compose_logit_adjustments(
    text: str,
    *,
    calibrator: KeywordIntentCalibrator | None = None,
    router: CognitiveIntentRouter | None = None,
) -> list[float]:
    vectors: list[list[float]] = []
    if calibrator is not None:
        vector = calibrator.vectorise(text)
        if vector:
            vectors.append(list(vector))
    if router is not None:
        vector = router.vectorise(text)
        if vector:
            vectors.append(list(vector))
    if not vectors:
        return []
    target_len = max(len(vector) for vector in vectors)
    combined = [0.0] * target_len
    for vector in vectors:
        if len(vector) < target_len:
            padded = vector + [0.0] * (target_len - len(vector))
        else:
            padded = vector[:target_len]
        for idx, value in enumerate(padded):
            combined[idx] += value
    return combined


def _stable_softmax(values: ndarray) -> ndarray:
    if values.size == 0:
        return values
    shifted = values - float(np.max(values))
    exp_values = np.exp(shifted)
    total = float(exp_values.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full_like(exp_values, fill_value=1.0 / max(len(exp_values), 1))
    return exp_values / total


def _is_sequence_like(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _coerce_snapshot_float(value: object | None, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, SupportsFloat):
        return float(value)
    return fallback


def _coerce_snapshot_int(value: object | None, fallback: int = 0) -> int:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, SupportsFloat):
        return int(float(value))
    return fallback


def _normalise_numeric_sequence(value: object) -> list[Any] | None:
    if not _is_sequence_like(value):
        return None
    normalised: list[Any] = []
    for item in value:
        if _is_sequence_like(item):
            nested = _normalise_numeric_sequence(item)
            if nested is None:
                return None
            normalised.append(nested)
        elif isinstance(item, (int, float)):
            normalised.append(float(item))
        elif isinstance(item, SupportsFloat):
            normalised.append(float(item))
        else:
            return None
    return normalised


def _snapshot_mapping(snapshot: object) -> dict[str, Any] | None:
    if not isinstance(snapshot, Mapping):
        return None
    typed: dict[str, Any] = {}
    items = cast(Iterable[tuple[object, object]], snapshot.items())
    for key_obj, value in items:
        if isinstance(key_obj, str):
            typed[key_obj] = value
    return typed


def _stringify_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    typed: dict[str, Any] = {}
    items = cast(Iterable[tuple[object, Any]], value.items())
    for key_obj, item_value in items:
        typed[str(key_obj)] = item_value
    return typed


def _ensure_tensor(
    value: object,
    *,
    device: TorchDevice | None = None,
    dtype: TorchDType | None = None,
    non_blocking: bool = False,
) -> TorchTensor:
    torch_module = torch
    if torch_module is None:
        raise RuntimeError("Torch is required to materialise tensors in this training pipeline.")
    if hasattr(torch_module, "is_tensor") and torch_module.is_tensor(value):
        tensor = cast(TorchTensor, value)
    else:
        tensor = torch_module.as_tensor(value)
    if device is not None or dtype is not None:
        tensor = tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)
    return tensor


def _normalise_history_item(entry: object) -> dict[str, float | int] | None:
    typed_entry = _snapshot_mapping(entry)
    if typed_entry is None:
        return None
    confident = _coerce_snapshot_int(typed_entry.get("confident"), 0)
    mean_confidence = _coerce_snapshot_float(typed_entry.get("mean_confidence"), 0.0)
    graph_entropy = _coerce_snapshot_float(typed_entry.get("graph_entropy"), 0.0)
    return {
        "confident": confident,
        "mean_confidence": mean_confidence,
        "graph_entropy": graph_entropy,
    }


def _metadata_section(metadata: object, key: str) -> Mapping[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    typed_metadata = cast(Mapping[str, Any], metadata)
    value = typed_metadata.get(key)
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return None


def _coerce_numpy_scalar(value: SupportsFloat | int | float | ndarray) -> float:
    """Convert a NumPy scalar or ndarray into a plain float for downstream maths."""

    if numpy_available and isinstance(value, ndarray):
        if value.size == 0:
            return 0.0
        scalar = value.item()
        if isinstance(scalar, (int, float)):
            return float(scalar)
        if isinstance(scalar, SupportsFloat):
            return float(scalar)
        # Fall back to NumPy's conversion semantics if the scalar has an unusual dtype.
        coerced = np.asarray(scalar, dtype=np.float64)
        if coerced.size > 0:
            coerced_scalar = coerced.reshape(-1).item()
            if isinstance(coerced_scalar, (int, float)):
                return float(coerced_scalar)
            if isinstance(coerced_scalar, SupportsFloat):
                return float(coerced_scalar)
            coerced_flat = coerced.reshape(-1)
            if coerced_flat.size > 0:
                return float(coerced_flat[0])
        return 0.0
    if isinstance(value, SupportsFloat):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Unsupported value for NumPy scalar coercion: {type(value)!r}")


class MetaIntentStacker:
    """Train a stacked meta-learner that refines logits via logistic calibration."""

    def __init__(
        self,
        *,
        label_to_idx: Mapping[str, int],
        scale: float = 0.85,
        regularization: float = 4.0,
        max_iter: int = 500,
        min_accuracy: float = 0.55,
    ) -> None:
        self.label_to_idx = dict(label_to_idx)
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        self.scale = float(max(scale, 0.0))
        self.regularization = float(max(regularization, 1e-4))
        self.max_iter = max(50, int(max_iter))
        self.min_accuracy = float(max(min_accuracy, 0.0))
        self.model: _LogisticModelProtocol | None = None
        self.scaler: _ScalerProtocol | None = None
        self.train_accuracy = 0.0
        self.trained_samples = 0
        self.feature_count = 0
        self.training_notes: str | None = None

    def _to_numpy(self, values: TorchTensor | Sequence[float] | None) -> ndarray:
        if values is None:
            return np.zeros(self.num_classes, dtype=np.float32)
        tensor_like: TorchTensor | None = None
        if hasattr(values, "detach") and hasattr(values, "cpu"):
            tensor_like = cast(TorchTensor, values)
        if tensor_like is not None:
            tensor_cpu = tensor_like.detach().cpu()
            array = tensor_cpu.numpy() if hasattr(tensor_cpu, "numpy") else np.asarray(tensor_cpu)
        else:
            array = np.asarray(values, dtype=np.float32)
        array = np.atleast_1d(array).astype(np.float32, copy=False)
        if array.size < self.num_classes:
            array = np.pad(array, (0, self.num_classes - array.size))
        elif array.size > self.num_classes:
            array = array[: self.num_classes]
        return array

    def _feature_vector(
        self,
        base_logits: TorchTensor | Sequence[float],
        keyword_logits: TorchTensor | Sequence[float] | None,
    ) -> ndarray:
        base: ndarray = self._to_numpy(base_logits)
        keyword: ndarray = self._to_numpy(keyword_logits)
        probs: ndarray = _stable_softmax(base)
        sorted_logits: ndarray = np.sort(base)
        top = _coerce_numpy_scalar(sorted_logits[-1]) if sorted_logits.size else 0.0
        second = (
            _coerce_numpy_scalar(sorted_logits[-2]) if sorted_logits.size >= 2 else top
        )
        base_std = _coerce_numpy_scalar(base.std()) if base.size else 0.0
        probs_max = _coerce_numpy_scalar(probs.max()) if probs.size else 0.0
        probs_min = _coerce_numpy_scalar(probs.min()) if probs.size else 0.0
        keyword_max = _coerce_numpy_scalar(keyword.max()) if keyword.size else 0.0
        keyword_min = _coerce_numpy_scalar(keyword.min()) if keyword.size else 0.0
        keyword_l1 = (
            _coerce_numpy_scalar(np.linalg.norm(keyword, ord=1)) if keyword.size else 0.0
        )
        keyword_l2 = (
            _coerce_numpy_scalar(np.linalg.norm(keyword, ord=2)) if keyword.size else 0.0
        )
        extras = np.array(
            [
                top,
                second,
                top - second if sorted_logits.size >= 2 else 0.0,
                base_std,
                probs_max,
                probs_min,
                keyword_max,
                keyword_min,
                keyword_l1,
                keyword_l2,
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, keyword, probs.astype(np.float32, copy=False), extras])

    def fit_from_dataloader(
        self,
        model: nn.Module,
        dataloader: DataLoader[BatchExample],
        device: TorchDevice,
        *,
        emotion_config: EmotionTrainingConfig | None = None,
    ) -> bool:
        if self.num_classes == 0:
            self.training_notes = "no_classes"
            return False
        non_blocking = device.type in {"cuda", "mps"}
        try:
            sklearn_linear = importlib.import_module("sklearn.linear_model")
            sklearn_preprocessing = importlib.import_module("sklearn.preprocessing")
        except ImportError:
            self.training_notes = "sklearn_unavailable"
            return False
        logistic_regression_cls = cast(
            _LogisticModelFactory, getattr(sklearn_linear, "LogisticRegression")
        )
        standard_scaler_cls = cast(
            _ScalerFactory, getattr(sklearn_preprocessing, "StandardScaler")
        )

        dataset_obj = getattr(dataloader, "dataset", None)
        dataset_has_emotion = bool(getattr(dataset_obj, "include_emotion", False))
        dataset_has_keywords = bool(getattr(dataset_obj, "include_keywords", False))
        features: list[ndarray] = []
        targets: list[int] = []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) < 4:
                    continue
                inputs = _ensure_tensor(batch[0], device=device, non_blocking=non_blocking)
                labels = _ensure_tensor(batch[1], device=device, non_blocking=non_blocking)
                attention_mask = _ensure_tensor(
                    batch[3], device=device, non_blocking=non_blocking
                )
                emotion_features: TorchTensor | None = None
                keyword_logits: TorchTensor | None = None
                next_index = 5
                if dataset_has_emotion and len(batch) > next_index:
                    emotion_candidate = batch[next_index]
                    if emotion_candidate is not None:
                        emotion_features = _ensure_tensor(
                            emotion_candidate,
                            device=device,
                            dtype=torch.float32,
                            non_blocking=non_blocking,
                        )
                        if emotion_features.dim() == 1:
                            emotion_features = emotion_features.unsqueeze(0)
                    next_index += 1
                if dataset_has_keywords and len(batch) > next_index:
                    keyword_candidate = batch[next_index]
                    if keyword_candidate is not None:
                        keyword_logits = _ensure_tensor(
                            keyword_candidate,
                            device=device,
                            dtype=torch.float32,
                            non_blocking=non_blocking,
                        )
                        if keyword_logits.dim() == 1:
                            keyword_logits = keyword_logits.unsqueeze(0)
                        if keyword_logits.numel() == 0:
                            keyword_logits = None
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
                for idx in range(logits.size(0)):
                    base_row = logits[idx]
                    keyword_row = None
                    if keyword_logits is not None:
                        tensor_keywords = keyword_logits
                        if idx < tensor_keywords.size(0):
                            keyword_row = tensor_keywords[idx]
                    features.append(self._feature_vector(base_row, keyword_row))
                    targets.append(int(labels[idx].item()))
        model.train(was_training)

        if not features or not targets:
            self.training_notes = "no_samples"
            return False
        unique_targets = set(targets)
        if len(unique_targets) < 2:
            self.training_notes = "insufficient_labels"
            return False

        X = np.stack(features, axis=0)
        y = np.asarray(targets, dtype=np.int64)
        scaler = standard_scaler_cls()
        X_scaled = scaler.fit_transform(X)
        clf = logistic_regression_cls(
            multi_class="ovr",
            solver="lbfgs",
            max_iter=self.max_iter,
            C=self.regularization,
            n_jobs=None,
        )
        clf.fit(X_scaled, y)
        preds = clf.predict(X_scaled)
        accuracy = float((preds == y).mean())
        self.train_accuracy = accuracy
        self.trained_samples = int(len(targets))
        self.feature_count = int(X.shape[1])
        if accuracy < self.min_accuracy:
            self.training_notes = "accuracy_threshold"
            return False
        self.scaler = scaler
        self.model = clf
        self.training_notes = None
        return True

    def compute_adjustment(
        self,
        base_logits: TorchTensor | Sequence[float],
        keyword_logits: TorchTensor | Sequence[float] | None = None,
    ) -> list[float] | None:
        model_impl = self.model
        scaler_impl = self.scaler
        if model_impl is None or scaler_impl is None or self.num_classes == 0:
            return None
        feature = self._feature_vector(base_logits, keyword_logits)
        feature = feature.reshape(1, -1)
        feature_scaled = scaler_impl.transform(feature)
        scores = model_impl.decision_function(feature_scaled)
        if scores.ndim == 1:
            if self.num_classes == 2:
                scores = np.stack([-scores, scores], axis=1)
            else:
                scores = scores.reshape(1, -1)
        if scores.shape[1] < self.num_classes:
            scores = np.pad(scores, ((0, 0), (0, self.num_classes - scores.shape[1])))
        adjustments = self.scale * scores[0]
        return adjustments.astype(np.float32, copy=False).tolist()

    def export_metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.model is not None),
            "scale": float(self.scale),
            "regularization": float(self.regularization),
            "max_iter": int(self.max_iter),
            "min_accuracy": float(self.min_accuracy),
            "trained_samples": int(self.trained_samples),
            "feature_count": int(self.feature_count),
            "training_accuracy": float(self.train_accuracy),
            "notes": self.training_notes,
        }

def build_keyword_intent_calibrator(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    label_to_idx: Mapping[str, int],
    min_frequency: int = 3,
    bigram_min_frequency: int = 2,
    smoothing: float = 0.5,
    strength_threshold: float = 0.1,
    max_features_per_label: int = 40,
    bias_weight: float = 0.75,
    feature_weight: float = 1.0,
    normalise_power: float = 0.5,
) -> KeywordIntentCalibrator | None:
    if not texts or not labels:
        return None
    if not label_to_idx:
        return None
    num_classes = len(label_to_idx)
    if num_classes == 0:
        return None
    min_frequency = max(1, int(min_frequency))
    bigram_min_frequency = max(1, int(bigram_min_frequency))
    smoothing = float(max(smoothing, 1e-6))
    max_features_per_label = max(1, int(max_features_per_label))

    label_counts = Counter(labels)
    total_examples = len(labels)
    baseline = 1.0 / num_classes
    bias: list[float] = []
    for label, idx in sorted(label_to_idx.items(), key=lambda item: item[1]):
        count = label_counts.get(label, 0)
        numerator = count + smoothing
        denominator = total_examples + smoothing * num_classes
        prob = max(numerator / max(denominator, 1e-8), 1e-8)
        bias.append(math.log(prob / baseline))

    label_feature_counts: dict[int, Counter[str]] = {
        idx: Counter() for idx in range(num_classes)
    }
    feature_document_counts: Counter[str] = Counter()
    for text, label in zip(texts, labels):
        idx = label_to_idx.get(label)
        if idx is None:
            continue
        tokens = tokenize(text)
        if not tokens:
            continue
        unigram_set = set(tokens)
        for token in unigram_set:
            if not token:
                continue
            label_feature_counts[idx][token] += 1
            feature_document_counts[token] += 1
        if len(tokens) >= 2:
            bigram_features = {
                f"{first}{BIGRAM_SEPARATOR}{second}"
                for first, second in zip(tokens, tokens[1:])
                if first and second
            }
            for feature in bigram_features:
                label_feature_counts[idx][feature] += 1
                feature_document_counts[feature] += 1

    feature_scores: dict[str, list[float]] = {}
    per_label_candidates: dict[int, list[tuple[str, float]]] = {
        idx: [] for idx in range(num_classes)
    }
    for feature, total_count in feature_document_counts.items():
        is_bigram = BIGRAM_SEPARATOR in feature
        threshold = bigram_min_frequency if is_bigram else min_frequency
        if total_count < threshold:
            continue
        counts = [label_feature_counts[idx][feature] for idx in range(num_classes)]
        feature_total = sum(counts)
        if feature_total < threshold:
            continue
        vector: list[float] = []
        denominator = feature_total + smoothing * num_classes
        for idx in range(num_classes):
            numerator = counts[idx] + smoothing
            prob = max(numerator / max(denominator, 1e-8), 1e-8)
            vector.append(math.log(prob / baseline))
        max_strength = max(abs(value) for value in vector)
        if max_strength < strength_threshold:
            continue
        feature_scores[feature] = vector
        for idx, score in enumerate(vector):
            if score > 0:
                per_label_candidates[idx].append((feature, score))

    keep_features: set[str] = set()
    top_features: dict[int, list[tuple[str, float]]] = {}
    for idx, candidates in per_label_candidates.items():
        if not candidates:
            continue
        sorted_candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
        selected = sorted_candidates[:max_features_per_label]
        top_features[idx] = selected
        keep_features.update(feature for feature, _ in selected)

    selected_scores = {
        feature: feature_scores[feature]
        for feature in keep_features
        if feature in feature_scores
    }
    if not selected_scores and not any(abs(value) > 1e-6 for value in bias):
        return None

    return KeywordIntentCalibrator(
        label_to_idx=label_to_idx,
        bias=bias,
        feature_scores=selected_scores,
        top_features=top_features,
        feature_weight=feature_weight,
        bias_weight=bias_weight,
        normalise_power=normalise_power,
        min_frequency=min_frequency,
        bigram_min_frequency=bigram_min_frequency,
        smoothing=smoothing,
        strength_threshold=strength_threshold,
    )


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

    def register_vector(self, label_idx: int, vector: TorchTensor, weight: float = 1.0) -> None:
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
        vectors: Sequence[Sequence[float] | TorchTensor],
        *,
        weights: Sequence[float] | None = None,
    ) -> None:
        if weights is None:
            weights = [1.0] * len(label_indices)
        for idx, vector, weight in zip(label_indices, vectors, weights):
            tensor_vec = (
                vector if isinstance(vector, TorchTensor) else torch.tensor(list(vector), dtype=torch.float32)
            )
            self.register_vector(idx, tensor_vec, float(weight))

    def register_texts(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        label_to_idx: dict[str, int],
        encoder: EmotionLexicon,
        weights: Sequence[float] | None = None,
    ) -> None:
        if weights is None:
            weights = [1.0] * len(texts)
        for text, label, weight in zip(texts, labels, weights):
            idx = label_to_idx.get(label)
            if idx is None:
                continue
            vector = torch.tensor(encoder.vectorise(text), dtype=torch.float32)
            self.register_vector(idx, vector, float(weight))

    def prototypes_tensor(self, *, device: TorchDevice | None = None, dtype: TorchDType | None = None) -> TorchTensor:
        denom = (self.prototype_weights.unsqueeze(1) + self.smoothing).clamp_min(1e-6)
        prototypes = self.prototype_sums / denom
        if device is not None or dtype is not None:
            prototypes = prototypes.to(device=device or prototypes.device, dtype=dtype or prototypes.dtype)
        return prototypes

    def alignment_loss(
        self,
        logits: TorchTensor,
        targets: TorchTensor,
        *,
        temperature: float = 1.0,
    ) -> TorchTensor:
        prototypes = self.prototypes_tensor(device=logits.device, dtype=logits.dtype)
        scaled_logits = logits / max(temperature, 1e-6)
        probs = torch.softmax(scaled_logits, dim=-1)
        expected = probs @ prototypes
        diff = expected - targets
        return (diff * diff).mean(dim=-1)

    def summary(self) -> dict[str, Any]:
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
    introspector: MetaCognitiveIntrospector
    enabled: bool
    attraction_weight: float
    repulsion_weight: float
    discovery_weight: float
    gap_margin: float
    temperature: float


@dataclass
class NeuroSymbolicConfig:
    reasoner: NeuroSymbolicReasoner
    enabled: bool
    structural_weight: float
    semantic_weight: float
    affective_weight: float
    temperature: float
    self_loop: float


@dataclass
class SelfDiscoveryConfig:
    orchestrator: SelfDiscoveryOrchestrator
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
    architect: TranscendentCognitionEngine
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
    catalyst: FrontierIntelligenceEngine
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
        self.history: deque[dict[str, Any]] = deque(maxlen=max(10, history_limit))
        self.feature_dim: int | None = None
        self.device: TorchDevice | None = None
        self.prototypes: TorchTensor | None = None
        self.prototype_counts: TorchTensor | None = None
        self.label_gap = torch.zeros(self.num_classes, dtype=torch.float32)
        self.label_entropy = torch.zeros(self.num_classes, dtype=torch.float32)
        self.avg_gap: float = 0.0
        self.avg_entropy: float = 0.0
        self.total_updates: int = 0

    def ensure_buffers(self, feature_dim: int, device: TorchDevice) -> None:
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
        features: TorchTensor,
        labels: TorchTensor,
        logits: TorchTensor,
        config: MetaCognitiveConfig,
    ) -> tuple[TorchTensor, dict[str, Any]]:
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

    def update_memory(self, features: TorchTensor, labels: TorchTensor, logits: TorchTensor) -> None:
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

    def snapshot(self) -> dict[str, Any]:
        prototypes_list: list[list[float]] | None = None
        counts_list: list[float] | None = None
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

    def load_snapshot(self, snapshot: dict[str, Any], device: TorchDevice) -> None:
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
        label_gap_raw = snapshot.get("label_gap")
        if isinstance(label_gap_raw, list):
            label_gap_items = cast(list[object], label_gap_raw)
            label_gap_values = [
                float(item)
                for item in label_gap_items
                if isinstance(item, (int, float))
            ]
            if len(label_gap_values) == self.num_classes:
                self.label_gap = torch.tensor(label_gap_values, dtype=torch.float32)
        label_entropy_raw = snapshot.get("label_entropy")
        if isinstance(label_entropy_raw, list):
            label_entropy_items = cast(list[object], label_entropy_raw)
            label_entropy_values = [
                float(item)
                for item in label_entropy_items
                if isinstance(item, (int, float))
            ]
            if len(label_entropy_values) == self.num_classes:
                self.label_entropy = torch.tensor(label_entropy_values, dtype=torch.float32)
        history_items_raw = snapshot.get("history", [])
        self.history.clear()
        if isinstance(history_items_raw, list):
            history_entries = cast(list[object], history_items_raw)
            history_items: list[Mapping[str, object]] = [
                cast(Mapping[str, object], entry)
                for entry in history_entries
                if isinstance(entry, Mapping)
            ]
            for item_mapping in history_items:
                gap_raw_obj = item_mapping.get("gap", 0.0)
                entropy_raw_obj = item_mapping.get("entropy", 0.0)
                confidence_raw_obj = item_mapping.get("confidence", 0.0)
                label_raw_obj = item_mapping.get("label")
                predicted_raw_obj = item_mapping.get("predicted")
                gap = float(gap_raw_obj) if isinstance(gap_raw_obj, (int, float)) else 0.0
                entropy = (
                    float(entropy_raw_obj) if isinstance(entropy_raw_obj, (int, float)) else 0.0
                )
                confidence = (
                    float(confidence_raw_obj)
                    if isinstance(confidence_raw_obj, (int, float))
                    else 0.0
                )
                label_value = str(label_raw_obj) if isinstance(label_raw_obj, str) else None
                predicted_value = (
                    str(predicted_raw_obj) if isinstance(predicted_raw_obj, str) else None
                )
                cleaned: dict[str, float | str | None] = {
                    "gap": gap,
                    "entropy": entropy,
                    "confidence": confidence,
                    "label": label_value,
                    "predicted": predicted_value,
                }
                self.history.append(cleaned)


class NeuroSymbolicReasoner:
    """Fuse lexical concept graphs with representation dynamics."""

    def __init__(
        self,
        num_classes: int,
        *,
        idx_to_label: Sequence[str],
        lexical_profiles: Sequence[dict[str, Any]],
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
        self.history: deque[dict[str, Any]] = deque(maxlen=max(10, history_limit))
        base_profiles = list(lexical_profiles)
        if len(base_profiles) < self.num_classes:
            base_profiles.extend({} for _ in range(self.num_classes - len(base_profiles)))
        self.lexical_matrix = self._build_lexical_matrix(base_profiles)
        self.feature_dim: int | None = None
        self.emotion_dim = int(max(0, emotion_dim))
        self.device: TorchDevice | None = None
        self.cooccurrence: TorchTensor | None = None
        self.graph_counts: TorchTensor | None = None
        self.concept_prototypes: TorchTensor | None = None
        self.concept_counts: TorchTensor | None = None
        self.emotion_prototypes: TorchTensor | None = None
        self.total_updates: int = 0
        self._pending_snapshot: dict[str, object] | None = None

    def _build_lexical_matrix(self, profiles: Sequence[dict[str, Any]]) -> TorchTensor:
        matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float32)
        norms: list[float] = []
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
            cooccurrence = _normalise_numeric_sequence(snapshot.get("cooccurrence"))
            if cooccurrence is not None:
                tensor = torch.tensor(cooccurrence, dtype=torch.float32, device=device)
                if tensor.shape == self.cooccurrence.shape:
                    self.cooccurrence.copy_(tensor)
        if self.graph_counts is not None:
            graph_counts = _normalise_numeric_sequence(snapshot.get("graph_counts"))
            if graph_counts is not None:
                tensor = torch.tensor(graph_counts, dtype=torch.float32, device=device)
                if tensor.shape == self.graph_counts.shape:
                    self.graph_counts.copy_(tensor)
        if self.concept_prototypes is not None:
            concept = _normalise_numeric_sequence(snapshot.get("concept_prototypes"))
            if concept is not None:
                tensor = torch.tensor(concept, dtype=torch.float32, device=device)
                if tensor.shape == self.concept_prototypes.shape:
                    self.concept_prototypes.copy_(tensor)
        if self.concept_counts is not None:
            concept_counts = _normalise_numeric_sequence(snapshot.get("concept_counts"))
            if concept_counts is not None:
                tensor = torch.tensor(concept_counts, dtype=torch.float32, device=device)
                if tensor.shape == self.concept_counts.shape:
                    self.concept_counts.copy_(tensor)
        emotion_proto = _normalise_numeric_sequence(snapshot.get("emotion_prototypes"))
        if emotion_proto is not None:
            tensor = torch.tensor(emotion_proto, dtype=torch.float32, device=device)
            if tensor.dim() == 2:
                self.emotion_dim = tensor.shape[1]
                emotion_prototypes = self.emotion_prototypes
                if emotion_prototypes is None or emotion_prototypes.shape != tensor.shape:
                    emotion_prototypes = torch.zeros_like(tensor)
                emotion_prototypes.copy_(tensor)
                self.emotion_prototypes = emotion_prototypes
        history_items = snapshot.get("history")
        if _is_sequence_like(history_items):
            for item in history_items:
                record = _normalise_history_item(item)
                if record is not None:
                    self.history.append(record)
        self.total_updates = _coerce_snapshot_int(
            snapshot.get("total_updates"),
            self.total_updates,
        )
        self._pending_snapshot = None

    def ensure_buffers(
        self,
        feature_dim: int,
        device: TorchDevice,
        emotion_dim: int | None = None,
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
        cooccurrence = self.cooccurrence
        graph_counts = self.graph_counts
        if cooccurrence is None:
            cooccurrence = torch.zeros(self.num_classes, self.num_classes, device=device)
            graph_counts = torch.zeros(self.num_classes, device=device)
        else:
            if cooccurrence.device != device:
                cooccurrence = cooccurrence.to(device)
            if graph_counts is not None and graph_counts.device != device:
                graph_counts = graph_counts.to(device)
        self.cooccurrence = cooccurrence
        self.graph_counts = graph_counts

        concept_prototypes = self.concept_prototypes
        concept_counts = self.concept_counts
        if concept_prototypes is None:
            concept_prototypes = torch.zeros(self.num_classes, feature_dim, device=device)
            concept_counts = torch.zeros(self.num_classes, device=device)
        else:
            if concept_prototypes.shape[1] != feature_dim:
                raise ValueError("Model feature dimension changed during neuro-symbolic reasoning.")
            if concept_prototypes.device != device:
                concept_prototypes = concept_prototypes.to(device)
            if concept_counts is not None and concept_counts.device != device:
                concept_counts = concept_counts.to(device)
        self.concept_prototypes = concept_prototypes
        self.concept_counts = concept_counts

        resolved_emotion_dim = emotion_dim if emotion_dim is not None else self.emotion_dim
        resolved_emotion_dim = int(max(0, resolved_emotion_dim))
        if resolved_emotion_dim > 0:
            self.emotion_dim = resolved_emotion_dim
            emotion_prototypes = self.emotion_prototypes
            if emotion_prototypes is None:
                emotion_prototypes = torch.zeros(self.num_classes, resolved_emotion_dim, device=device)
            else:
                if emotion_prototypes.shape[1] != resolved_emotion_dim:
                    emotion_prototypes = torch.zeros(
                        self.num_classes,
                        resolved_emotion_dim,
                        device=device,
                    )
                elif emotion_prototypes.device != device:
                    emotion_prototypes = emotion_prototypes.to(device)
            self.emotion_prototypes = emotion_prototypes
        else:
            if emotion_dim is not None:
                self.emotion_prototypes = None
                self.emotion_dim = 0
            elif self.emotion_prototypes is not None and self.emotion_prototypes.device != device:
                self.emotion_prototypes = self.emotion_prototypes.to(device)
        self.lexical_matrix = self.lexical_matrix.to(device)
        self._apply_pending_snapshot()

    def adjacency_matrix(self, device: TorchDevice | None = None) -> TorchTensor:
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
        features: TorchTensor,
        logits: TorchTensor,
        labels: TorchTensor,
        *,
        emotion_features: TorchTensor | None,
        config: NeuroSymbolicConfig,
    ) -> tuple[TorchTensor, dict[str, Any]]:
        if features.dim() != 2:
            features = features.view(features.size(0), -1)
        emotion_dim = None
        has_emotion = emotion_features is not None and emotion_features.numel() > 0
        if has_emotion and emotion_features is not None:
            emo_tensor = emotion_features
            if emo_tensor.dim() == 1:
                emo_tensor = emo_tensor.unsqueeze(0)
            emotion_dim = emo_tensor.shape[-1]
            emotion_features = emo_tensor
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
        concept_prototypes = self.concept_prototypes
        if concept_prototypes is not None:
            semantic_targets = target_distribution @ concept_prototypes
            semantic = (features - semantic_targets).pow(2).sum(dim=1)
        affective = torch.zeros_like(structural)
        emotion_prototypes = self.emotion_prototypes
        if has_emotion and emotion_features is not None and emotion_prototypes is not None:
            affective_targets = target_distribution @ emotion_prototypes
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
        features: TorchTensor,
        logits: TorchTensor,
        labels: TorchTensor,
        *,
        emotion_features: TorchTensor | None,
    ) -> None:
        if features.numel() == 0:
            return
        if features.dim() != 2:
            features = features.view(features.size(0), -1)
        emotion_dim = None
        has_emotion = emotion_features is not None and emotion_features.numel() > 0
        if has_emotion and emotion_features is not None:
            emo_tensor = emotion_features
            if emo_tensor.dim() == 1:
                emo_tensor = emo_tensor.unsqueeze(0)
            emotion_dim = emo_tensor.shape[-1]
            emotion_features = emo_tensor
        self.ensure_buffers(features.shape[1], features.device, emotion_dim)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            confidences, _ = probs.max(dim=1)
            confident_mask = confidences >= self.min_confidence
            cooccurrence = self.cooccurrence
            graph_counts = self.graph_counts
            if confident_mask.any() and cooccurrence is not None and graph_counts is not None:
                confident_probs = probs[confident_mask]
                confident_labels = labels[confident_mask]
                one_hot = F.one_hot(confident_labels, num_classes=self.num_classes).float()
                cooc_update = one_hot.T @ confident_probs
                self.cooccurrence = (
                    (1 - self.graph_momentum) * cooccurrence
                    + self.graph_momentum * cooc_update
                )
                count_update = one_hot.sum(dim=0)
                self.graph_counts = (
                    (1 - self.graph_momentum) * graph_counts
                    + self.graph_momentum * count_update
                )

            class_counts = torch.bincount(labels, minlength=self.num_classes).float().to(features.device)
            concept_prototypes = self.concept_prototypes
            concept_counts = self.concept_counts
            if class_counts.sum() > 0 and concept_prototypes is not None and concept_counts is not None:
                feature_sums = torch.zeros(self.num_classes, features.shape[1], device=features.device)
                feature_sums.index_add_(0, labels, features)
                for idx in range(self.num_classes):
                    count = float(class_counts[idx].item())
                    if count <= 0:
                        continue
                    mean_vec = feature_sums[idx] / max(count, 1.0)
                    if float(concept_counts[idx].item()) <= 0:
                        concept_prototypes[idx] = mean_vec
                        concept_counts[idx] = class_counts[idx]
                    else:
                        blend = self.feature_momentum
                        concept_prototypes[idx] = (
                            (1 - blend) * concept_prototypes[idx] + blend * mean_vec
                        )
                        concept_counts[idx] = (
                            (1 - blend) * concept_counts[idx] + blend * class_counts[idx]
                        )
                self.concept_prototypes = concept_prototypes
                self.concept_counts = concept_counts

            emotion_prototypes = self.emotion_prototypes
            if has_emotion and emotion_features is not None and emotion_prototypes is not None:
                emotion_sums = torch.zeros(self.num_classes, emotion_prototypes.shape[1], device=features.device)
                emotion_sums.index_add_(0, labels, emotion_features)
                for idx in range(self.num_classes):
                    count = float(class_counts[idx].item())
                    if count <= 0:
                        continue
                    mean_vec = emotion_sums[idx] / max(count, 1.0)
                    current = emotion_prototypes[idx]
                    if float(current.abs().sum().item()) <= 0:
                        emotion_prototypes[idx] = mean_vec
                    else:
                        blend = self.feature_momentum
                        emotion_prototypes[idx] = (
                            (1 - blend) * current + blend * mean_vec
                        )
                self.emotion_prototypes = emotion_prototypes

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

    def export_metadata(self) -> dict[str, Any]:
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

    def export_metrics(self) -> dict[str, Any]:
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

    def snapshot(self) -> dict[str, Any]:
        return {
            "cooccurrence": self.cooccurrence.detach().cpu().tolist() if self.cooccurrence is not None else None,
            "graph_counts": self.graph_counts.detach().cpu().tolist() if self.graph_counts is not None else None,
            "concept_prototypes": self.concept_prototypes.detach().cpu().tolist() if self.concept_prototypes is not None else None,
            "concept_counts": self.concept_counts.detach().cpu().tolist() if self.concept_counts is not None else None,
            "emotion_prototypes": self.emotion_prototypes.detach().cpu().tolist() if self.emotion_prototypes is not None else None,
            "history": list(self.history),
            "total_updates": int(self.total_updates),
        }

    def load_snapshot(self, snapshot: object) -> None:
        typed_snapshot = _snapshot_mapping(snapshot)
        if typed_snapshot is None:
            return
        self._pending_snapshot = typed_snapshot


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
        self.history: deque[dict[str, Any]] = deque(maxlen=max(16, history_limit))
        self.feature_dim: int | None = None
        self.emotion_dim: int = 0
        self.device: TorchDevice | None = None
        self.positive_prototypes: TorchTensor | None = None
        self.positive_counts: TorchTensor | None = None
        self.counterfactual_prototypes: TorchTensor | None = None
        self.counterfactual_counts: TorchTensor | None = None
        self.expectation_trace: TorchTensor | None = None
        self.expectation_counts: TorchTensor | None = None
        self.curiosity_trace: TorchTensor | None = None
        self.emotion_positive: TorchTensor | None = None
        self.emotion_counter: TorchTensor | None = None
        self.total_updates: int = 0
        self._pending_snapshot: dict[str, object] | None = None

    def ensure_buffers(
        self,
        feature_dim: int,
        device: TorchDevice,
        emotion_dim: int | None = None,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive for self-discovery orchestration")
        positive_prototypes = self.positive_prototypes
        needs_init = positive_prototypes is None or positive_prototypes.shape[1] != feature_dim
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
        elif positive_prototypes is not None and positive_prototypes.device != device:
            self.positive_prototypes = positive_prototypes.to(device)
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
            emotion_positive = self.emotion_positive
            emotion_counter = self.emotion_counter
            if (
                emotion_positive is None
                or emotion_positive.shape[1] != resolved_emotion_dim
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
            elif emotion_positive.device != device:
                self.emotion_positive = emotion_positive.to(device)
                if emotion_counter is not None:
                    self.emotion_counter = emotion_counter.to(device)
            self.emotion_dim = resolved_emotion_dim
        self._apply_pending_snapshot()

    def _apply_pending_snapshot(self) -> None:
        if not self._pending_snapshot or self.device is None or self.feature_dim is None:
            return
        snapshot = self._pending_snapshot
        device = self.device
        try:
            positive = _normalise_numeric_sequence(snapshot.get("positive_prototypes"))
            if positive is not None and self.positive_prototypes is not None:
                tensor = torch.tensor(positive, dtype=torch.float32, device=device)
                if tensor.shape == self.positive_prototypes.shape:
                    self.positive_prototypes.copy_(tensor)
            positive_counts = _normalise_numeric_sequence(snapshot.get("positive_counts"))
            if positive_counts is not None and self.positive_counts is not None:
                tensor = torch.tensor(positive_counts, dtype=torch.float32, device=device)
                if tensor.shape == self.positive_counts.shape:
                    self.positive_counts.copy_(tensor)
            counter = _normalise_numeric_sequence(snapshot.get("counterfactual_prototypes"))
            if counter is not None and self.counterfactual_prototypes is not None:
                tensor = torch.tensor(counter, dtype=torch.float32, device=device)
                if tensor.shape == self.counterfactual_prototypes.shape:
                    self.counterfactual_prototypes.copy_(tensor)
            counter_counts = _normalise_numeric_sequence(snapshot.get("counterfactual_counts"))
            if counter_counts is not None and self.counterfactual_counts is not None:
                tensor = torch.tensor(counter_counts, dtype=torch.float32, device=device)
                if tensor.shape == self.counterfactual_counts.shape:
                    self.counterfactual_counts.copy_(tensor)
            expectation = _normalise_numeric_sequence(snapshot.get("expectation_trace"))
            if expectation is not None and self.expectation_trace is not None:
                tensor = torch.tensor(expectation, dtype=torch.float32, device=device)
                if tensor.shape == self.expectation_trace.shape:
                    self.expectation_trace.copy_(tensor)
            expectation_counts = _normalise_numeric_sequence(snapshot.get("expectation_counts"))
            if expectation_counts is not None and self.expectation_counts is not None:
                tensor = torch.tensor(expectation_counts, dtype=torch.float32, device=device)
                if tensor.shape == self.expectation_counts.shape:
                    self.expectation_counts.copy_(tensor)
            curiosity_trace = _normalise_numeric_sequence(snapshot.get("curiosity_trace"))
            if curiosity_trace is not None and self.curiosity_trace is not None:
                tensor = torch.tensor(curiosity_trace, dtype=torch.float32, device=device)
                if tensor.shape == self.curiosity_trace.shape:
                    self.curiosity_trace.copy_(tensor)
            emotion_positive = _normalise_numeric_sequence(snapshot.get("emotion_positive"))
            if emotion_positive is not None and self.emotion_positive is not None:
                tensor = torch.tensor(emotion_positive, dtype=torch.float32, device=device)
                if tensor.shape == self.emotion_positive.shape:
                    self.emotion_positive.copy_(tensor)
            emotion_counter = _normalise_numeric_sequence(snapshot.get("emotion_counter"))
            if emotion_counter is not None and self.emotion_counter is not None:
                tensor = torch.tensor(emotion_counter, dtype=torch.float32, device=device)
                if tensor.shape == self.emotion_counter.shape:
                    self.emotion_counter.copy_(tensor)
        finally:
            self.total_updates = _coerce_snapshot_int(
                snapshot.get("total_updates"),
                self.total_updates,
            )
            history_items = snapshot.get("history")
            self.history.clear()
            if _is_sequence_like(history_items):
                for item in history_items:
                    typed_item = _snapshot_mapping(item)
                    if typed_item is None:
                        continue
                    cleaned = {
                        "confidence": _coerce_snapshot_float(typed_item.get("confidence"), 0.0),
                        "curiosity": _coerce_snapshot_float(typed_item.get("curiosity"), 0.0),
                        "counter_examples": _coerce_snapshot_int(
                            typed_item.get("counter_examples"),
                            0,
                        ),
                    }
                    self.history.append(cleaned)
            self._pending_snapshot = None

    def compute_loss(
        self,
        features: TorchTensor,
        logits: TorchTensor,
        labels: TorchTensor,
        *,
        config: SelfDiscoveryConfig,
        emotion_features: TorchTensor | None = None,
    ) -> tuple[TorchTensor, dict[str, Any]]:
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
        features: TorchTensor,
        logits: TorchTensor,
        labels: TorchTensor,
        *,
        config: SelfDiscoveryConfig,
        emotion_features: TorchTensor | None = None,
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
            counter_mask_active = bool(needs_counter.any())
            hard_features: TorchTensor | None = None
            hard_labels: TorchTensor | None = None
            hard_pred: TorchTensor | None = None
            if counter_mask_active:
                hard_features = features[needs_counter]
                hard_labels = labels[needs_counter]
                hard_pred = predicted[needs_counter]
                assert hard_features is not None and hard_labels is not None and hard_pred is not None
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
                if counter_mask_active and self.emotion_counter is not None and hard_labels is not None and hard_pred is not None:
                    hard_emotion = emotion_features[needs_counter]
                    assert hard_emotion is not None
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

    def snapshot(self) -> dict[str, Any]:
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

    def load_snapshot(self, snapshot: object) -> None:
        typed_snapshot = _snapshot_mapping(snapshot)
        if typed_snapshot is None:
            return
        self._pending_snapshot = typed_snapshot
        if self.device is not None and self.feature_dim is not None:
            self._apply_pending_snapshot()

    def export_metadata(self) -> dict[str, Any]:
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
        self.history: deque[dict[str, Any]] = deque(maxlen=max(32, history_limit))
        self.feature_dim: int | None = None
        self.emotion_dim: int = 0
        self.device: TorchDevice | None = None
        self.feature_bank: TorchTensor | None = None
        self.counter_bank: TorchTensor | None = None
        self.imagination_bank: TorchTensor | None = None
        self.transition_matrix: TorchTensor | None = None
        self.emotion_bank: TorchTensor | None = None
        self.total_updates: int = 0
        self._pending_snapshot: dict[str, object] | None = None

    def _trim_distribution(self, distribution: TorchTensor) -> TorchTensor:
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

        def _copy_tensor(attr: TorchTensor | None, key: str) -> TorchTensor | None:
            if attr is None:
                return None
            data = _normalise_numeric_sequence(snapshot.get(key))
            if data is None:
                return attr
            tensor = torch.tensor(data, dtype=torch.float32, device=device)
            if tensor.shape == attr.shape:
                attr.copy_(tensor)
            return attr

        self.feature_bank = _copy_tensor(self.feature_bank, "feature_bank")
        self.counter_bank = _copy_tensor(self.counter_bank, "counter_bank")
        self.imagination_bank = _copy_tensor(self.imagination_bank, "imagination_bank")
        transition = self.transition_matrix
        if transition is not None:
            transition_data = _normalise_numeric_sequence(snapshot.get("transition_matrix"))
            if transition_data is not None:
                tensor = torch.tensor(transition_data, dtype=torch.float32, device=device)
                if tensor.shape == transition.shape:
                    transition.copy_(tensor)
                    self.transition_matrix = transition
        emotion_data = _normalise_numeric_sequence(snapshot.get("emotion_bank"))
        if emotion_data is not None:
            tensor = torch.tensor(emotion_data, dtype=torch.float32, device=device)
            if tensor.dim() == 2:
                self.emotion_dim = tensor.shape[1]
                emotion_bank = self.emotion_bank
                if emotion_bank is None or emotion_bank.shape != tensor.shape:
                    emotion_bank = torch.zeros_like(tensor)
                emotion_bank.copy_(tensor)
                self.emotion_bank = emotion_bank
        history_items = snapshot.get("history")
        if _is_sequence_like(history_items):
            for item in history_items:
                typed_item = _snapshot_mapping(item)
                if typed_item is None:
                    continue
                entry = {
                    "confidence": _coerce_snapshot_float(typed_item.get("confidence"), 0.0),
                    "entropy": _coerce_snapshot_float(typed_item.get("entropy"), 0.0),
                    "transition_coherence": _coerce_snapshot_float(
                        typed_item.get("transition_coherence"),
                        0.0,
                    ),
                }
                self.history.append(entry)
        self.total_updates = _coerce_snapshot_int(
            snapshot.get("total_updates"),
            self.total_updates,
        )
        self._pending_snapshot = None

    def ensure_buffers(
        self,
        feature_dim: int,
        device: TorchDevice,
        emotion_dim: int | None = None,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive for transcendent cognition")
        feature_bank = self.feature_bank
        counter_bank = self.counter_bank
        imagination_bank = self.imagination_bank
        needs_init = feature_bank is None or feature_bank.shape[1] != feature_dim
        if needs_init:
            self.feature_dim = feature_dim
            self.device = device
            feature_bank = torch.zeros(self.num_classes, feature_dim, device=device)
            counter_bank = torch.zeros(self.num_classes, feature_dim, device=device)
            imagination_bank = torch.zeros(self.num_classes, feature_dim, device=device)
        else:
            if feature_bank is not None and feature_bank.device != device:
                feature_bank = feature_bank.to(device)
            if counter_bank is not None and counter_bank.device != device:
                counter_bank = counter_bank.to(device)
            if imagination_bank is not None and imagination_bank.device != device:
                imagination_bank = imagination_bank.to(device)
            self.device = device
        self.feature_bank = feature_bank
        self.counter_bank = counter_bank
        self.imagination_bank = imagination_bank

        transition_matrix = self.transition_matrix
        if transition_matrix is None or transition_matrix.shape != (self.num_classes, self.num_classes):
            transition_matrix = torch.eye(self.num_classes, device=device)
        elif transition_matrix.device != device:
            transition_matrix = transition_matrix.to(device)
        self.transition_matrix = transition_matrix

        if emotion_dim is not None and emotion_dim > 0:
            emotion_bank = self.emotion_bank
            if emotion_bank is None or emotion_bank.shape[1] != emotion_dim:
                self.emotion_dim = emotion_dim
                emotion_bank = torch.zeros(self.num_classes, emotion_dim, device=device)
            elif emotion_bank.device != device:
                emotion_bank = emotion_bank.to(device)
            self.emotion_bank = emotion_bank
        else:
            if self.emotion_bank is not None and self.emotion_bank.device != device:
                self.emotion_bank = self.emotion_bank.to(device)
        self._apply_pending_snapshot()

    def normalised_transition(self, *, device: TorchDevice | None = None) -> TorchTensor:
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
        features: TorchTensor,
        logits: TorchTensor,
        targets: TorchTensor,
        config: TranscendentCognitionConfig,
        *,
        emotion_features: TorchTensor | None = None,
    ) -> tuple[TorchTensor, dict[str, Any]]:
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
        if (
            self.feature_bank is None
            or self.counter_bank is None
            or self.transition_matrix is None
            or self.imagination_bank is None
        ):
            raise RuntimeError("TranscendentCognitionEngine buffers failed to initialise.")

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
        features: TorchTensor,
        logits: TorchTensor,
        targets: TorchTensor,
        *,
        config: TranscendentCognitionConfig,
        emotion_features: TorchTensor | None = None,
    ) -> None:
        if features.numel() == 0:
            return
        emotion_dim = None
        if emotion_features is not None and emotion_features.numel() > 0 and emotion_features.dim() == 2:
            emotion_dim = emotion_features.shape[1]
        self.ensure_buffers(features.shape[1], features.device, emotion_dim)
        feature_bank = self.feature_bank
        counter_bank = self.counter_bank
        imagination_bank = self.imagination_bank
        transition_matrix = self.transition_matrix
        if (
            feature_bank is None
            or counter_bank is None
            or imagination_bank is None
            or transition_matrix is None
        ):
            raise RuntimeError("TranscendentCognitionEngine buffers failed to initialise.")
        emotion_bank = self.emotion_bank

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
                feature_bank[idx] = (1 - blend) * feature_bank[idx] + blend * class_mean

            expected = distribution.t() @ features
            counts = distribution.sum(dim=0).unsqueeze(1)
            for idx in range(self.num_classes):
                count = float(counts[idx].item())
                if count <= 0:
                    continue
                target_mean = expected[idx] / max(count, 1e-6)
                update = target_mean - feature_bank[idx]
                c_blend = self.counter_momentum
                counter_bank[idx] = (1 - c_blend) * counter_bank[idx] + c_blend * update

            t_blend = self.transition_momentum
            for class_idx in targets.unique(sorted=False):
                idx = int(class_idx.item())
                mask = targets == class_idx
                if not mask.any():
                    continue
                mean_trans = distribution[mask].mean(dim=0)
                transition_matrix[idx] = (1 - t_blend) * transition_matrix[idx] + t_blend * mean_trans

            combined = feature_bank + counter_bank
            imagined = distribution @ combined
            i_blend = self.imagination_momentum
            for class_idx in targets.unique(sorted=False):
                idx = int(class_idx.item())
                mask = targets == class_idx
                if not mask.any():
                    continue
                imagined_mean = imagined[mask].mean(dim=0)
                imagination_bank[idx] = (1 - i_blend) * imagination_bank[idx] + i_blend * imagined_mean

            if (
                emotion_features is not None
                and emotion_features.numel() > 0
                  and emotion_bank is not None
            ):
                emotion_expected = distribution.t() @ emotion_features
                emotion_counts = distribution.sum(dim=0).unsqueeze(1)
                for idx in range(self.num_classes):
                    count = float(emotion_counts[idx].item())
                    if count <= 0:
                        continue
                    update = emotion_expected[idx] / max(count, 1e-6)
                    e_blend = self.counter_momentum
                    emotion_bank[idx] = (1 - e_blend) * emotion_bank[idx] + e_blend * update

            entropy = -(distribution.clamp_min(1e-9).log() * distribution).sum(dim=1)
            confidence, _ = distribution.max(dim=1)
            coherence = float(torch.diagonal(self.normalised_transition()).mean().item())
            self.history.append(
                {
                    "confidence": float(confidence.mean().item()),
                    "entropy": float(entropy.mean().item()),
                    "transition_coherence": coherence,
                }
            )
            self.total_updates += int(features.shape[0])

    def export_metadata(self) -> dict[str, Any]:
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

    def export_metrics(self) -> dict[str, Any]:
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

    def snapshot(self) -> dict[str, Any]:
        return {
            "feature_bank": self.feature_bank.detach().cpu().tolist() if self.feature_bank is not None else None,
            "counter_bank": self.counter_bank.detach().cpu().tolist() if self.counter_bank is not None else None,
            "imagination_bank": self.imagination_bank.detach().cpu().tolist() if self.imagination_bank is not None else None,
            "transition_matrix": self.transition_matrix.detach().cpu().tolist() if self.transition_matrix is not None else None,
            "emotion_bank": self.emotion_bank.detach().cpu().tolist() if self.emotion_bank is not None else None,
            "history": list(self.history),
            "total_updates": int(self.total_updates),
        }

    def load_snapshot(self, snapshot: object) -> None:
        typed_snapshot = _snapshot_mapping(snapshot)
        if typed_snapshot is None:
            return
        self._pending_snapshot = typed_snapshot
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
        self.history: deque[dict[str, Any]] = deque(maxlen=max(32, history_limit))
        self.feature_dim: int | None = None
        self.emotion_dim: int = 0
        self.device: TorchDevice | None = None
        self.concept_bank: TorchTensor | None = None
        self.bridge_bank: TorchTensor | None = None
        self.novelty_bank: TorchTensor | None = None
        self.meta_bias: TorchTensor | None = None
        self.emotion_bank: TorchTensor | None = None
        self.total_updates: int = 0
        self._pending_snapshot: dict[str, object] | None = None

    def ensure_buffers(
        self,
        feature_dim: int,
        device: TorchDevice,
        emotion_dim: int | None = None,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive for frontier intelligence")
        concept_bank = self.concept_bank
        needs_init = concept_bank is None or concept_bank.shape[1] != feature_dim
        if needs_init:
            self.feature_dim = feature_dim
            self.device = device
            self.concept_bank = torch.zeros(self.num_classes, feature_dim, device=device)
            self.bridge_bank = torch.zeros(self.num_classes, feature_dim, device=device)
            self.novelty_bank = torch.zeros(self.num_classes, feature_dim, device=device)
            self.meta_bias = torch.zeros(feature_dim, device=device)
        elif concept_bank is not None and concept_bank.device != device:
            self.concept_bank = concept_bank.to(device)
            if self.bridge_bank is not None:
                self.bridge_bank = self.bridge_bank.to(device)
            if self.novelty_bank is not None:
                self.novelty_bank = self.novelty_bank.to(device)
            if self.meta_bias is not None:
                self.meta_bias = self.meta_bias.to(device)
            self.device = device
        resolved_emotion_dim = int(emotion_dim or self.emotion_dim or 0)
        if resolved_emotion_dim > 0:
            emotion_bank = self.emotion_bank
            if (
                emotion_bank is None
                or emotion_bank.shape[1] != resolved_emotion_dim
            ):
                self.emotion_bank = torch.zeros(
                    self.num_classes,
                    resolved_emotion_dim,
                    device=device,
                )
            elif emotion_bank.device != device:
                self.emotion_bank = emotion_bank.to(device)
            self.emotion_dim = resolved_emotion_dim
        self._apply_pending_snapshot()

    def _apply_pending_snapshot(self) -> None:
        if not self._pending_snapshot or self.device is None or self.feature_dim is None:
            return
        snapshot = self._pending_snapshot
        device = self.device

        def _copy_tensor(current: TorchTensor | None, key: str) -> TorchTensor | None:
            data = _normalise_numeric_sequence(snapshot.get(key))
            if current is None or data is None:
                return current
            tensor = torch.tensor(data, dtype=torch.float32, device=device)
            if tensor.shape == current.shape:
                current.copy_(tensor)
            return current

        self.concept_bank = _copy_tensor(self.concept_bank, "concept_bank")
        self.bridge_bank = _copy_tensor(self.bridge_bank, "bridge_bank")
        self.novelty_bank = _copy_tensor(self.novelty_bank, "novelty_bank")
        if self.meta_bias is not None:
            meta_data = _normalise_numeric_sequence(snapshot.get("meta_bias"))
            if meta_data is not None:
                tensor = torch.tensor(meta_data, dtype=torch.float32, device=device)
                if tensor.shape == self.meta_bias.shape:
                    self.meta_bias.copy_(tensor)
        if self.emotion_bank is not None:
            self.emotion_bank = _copy_tensor(self.emotion_bank, "emotion_bank")
        history_items = snapshot.get("history")
        if _is_sequence_like(history_items):
            for item in history_items:
                typed_item = _snapshot_mapping(item)
                if typed_item is None:
                    continue
                self.history.append(
                    {
                        "confidence": _coerce_snapshot_float(typed_item.get("confidence"), 0.0),
                        "curiosity": _coerce_snapshot_float(typed_item.get("curiosity"), 0.0),
                        "diversity": _coerce_snapshot_float(typed_item.get("diversity"), 0.0),
                    }
                )
        self.total_updates = _coerce_snapshot_int(
            snapshot.get("total_updates"),
            self.total_updates,
        )
        self._pending_snapshot = None

    def compute_loss(
        self,
        features: TorchTensor,
        logits: TorchTensor,
        targets: TorchTensor,
        config: FrontierIntelligenceConfig,
        *,
        emotion_features: TorchTensor | None = None,
    ) -> tuple[TorchTensor, dict[str, Any]]:
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

        abstraction_target = distribution @ self.concept_bank
        transfer_target = distribution @ self.bridge_bank

        novelty_distance = torch.cdist(features, self.novelty_bank)
        novelty_values, _ = novelty_distance.min(dim=1)
        novelty_gap = torch.relu(config.margin - novelty_values)
        abstraction_penalty = (features - abstraction_target).pow(2).sum(dim=1)
        transfer_penalty = (features - transfer_target).pow(2).sum(dim=1)
        meta_penalty = (features - self.meta_bias.unsqueeze(0)).pow(2).sum(dim=1)
        confidence, _ = distribution.max(dim=1)
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
        features: TorchTensor,
        logits: TorchTensor,
        targets: TorchTensor,
        *,
        config: FrontierIntelligenceConfig,
        emotion_features: TorchTensor | None = None,
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
        confidence, _ = distribution.max(dim=1)
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

    def export_metadata(self) -> dict[str, Any]:
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

    def export_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
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

    def snapshot(self) -> dict[str, Any]:
        return {
            "concept_bank": self.concept_bank.detach().cpu().tolist() if self.concept_bank is not None else None,
            "bridge_bank": self.bridge_bank.detach().cpu().tolist() if self.bridge_bank is not None else None,
            "novelty_bank": self.novelty_bank.detach().cpu().tolist() if self.novelty_bank is not None else None,
            "meta_bias": self.meta_bias.detach().cpu().tolist() if self.meta_bias is not None else None,
            "emotion_bank": self.emotion_bank.detach().cpu().tolist() if self.emotion_bank is not None else None,
            "history": list(self.history),
            "total_updates": int(self.total_updates),
        }

    def load_snapshot(self, snapshot: object) -> None:
        typed_snapshot = _snapshot_mapping(snapshot)
        if typed_snapshot is None:
            return
        self._pending_snapshot = typed_snapshot
        if self.device is not None and self.feature_dim is not None:
            self._apply_pending_snapshot()


@dataclass
class EncodedExample:
    tokens: TorchTensor
    attention_mask: TorchTensor
    label: int
    weight: float
    teacher_logits: TorchTensor
    emotion_vector: TorchTensor
    keyword_logits: TorchTensor


@dataclass
class FoldResult:
    fold_index: int
    val_accuracy: float
    train_accuracy_at_best: float
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    history: list[dict[str, Any]]
    pseudo_rounds: list[dict[str, Any]]
    total_pseudo_added: int
    pseudo_examples: list[dict[str, Any]]
    self_play_rounds: list[dict[str, Any]]
    total_self_play_added: int
    self_play_examples: list[tuple[str, str, float]]
    model_state: dict[str, Any]
    evaluation_outputs: list[dict[str, str | float]]
    run_tag_suffix: str | None


@dataclass
class ClassBalanceConfig:
    strategy: str
    boost: float
    power: float
    momentum: float
    min_multiplier: float
    max_multiplier: float
    floor: float
    min_support: int


class ClassWeightBalancer:
    """Dynamically reweight classes based on validation performance."""

    def __init__(self, config: ClassBalanceConfig, labels: Sequence[str]) -> None:
        self.config = config
        self.enabled = config.strategy != "none"
        self.label_counts: dict[str, int] = Counter(labels)
        self.multipliers: dict[str, Any] = {
            label: 1.0 for label in self.label_counts
        }
        self.applied: dict[str, Any] = {
            label: 1.0 for label in self.label_counts
        }
        self.last_metrics: dict[str, dict[str, Any]] = {}

    def register_samples(self, labels: Sequence[str]) -> None:
        if not labels:
            return
        for label in labels:
            self.label_counts[label] = self.label_counts.get(label, 0) + 1
            if label not in self.multipliers:
                self.multipliers[label] = 1.0
            if label not in self.applied:
                self.applied[label] = 1.0

    def last_applied(self, label: str) -> float:
        return self.applied.get(label, 1.0)

    def current_multiplier(self, label: str) -> float:
        return self.multipliers.get(label, 1.0)

    def update(self, metrics: dict[str, dict[str, Any]]) -> None:
        if not self.enabled:
            return
        self.last_metrics = metrics
        observed_labels = set(self.multipliers) | set(metrics)
        for label in observed_labels:
            stats = metrics.get(label)
            if stats is None:
                target_error = 0.0
            else:
                recall = float(stats.get("recall", 0.0))
                precision = float(stats.get("precision", recall))
                if self.config.strategy == "recall":
                    target_error = max(0.0, 1.0 - recall)
                elif self.config.strategy == "f1":
                    f1 = float(stats.get("f1", 0.0))
                    target_error = max(0.0, 1.0 - f1)
                else:
                    # blended precision/recall emphasis
                    target_error = max(0.0, 1.0 - 0.5 * (recall + precision))
                support = float(stats.get("support", self.label_counts.get(label, 0)))
                if support < self.config.min_support:
                    scarcity = support / max(1.0, float(self.config.min_support))
                    target_error *= scarcity
            scaled = 1.0 + (target_error ** self.config.power) * self.config.boost
            previous = self.multipliers.get(label, 1.0)
            updated = previous * self.config.momentum + scaled * (1.0 - self.config.momentum)
            updated = max(self.config.min_multiplier, min(self.config.max_multiplier, updated))
            if updated < self.config.floor:
                updated = self.config.floor
            self.multipliers[label] = updated

    def apply(
        self,
        labels: Sequence[str],
        weights: MutableSequence[float],
        *,
        dataset_examples: Sequence[EncodedExample] | None = None,
        idx_to_label: Sequence[str] | None = None,
    ) -> None:
        if not self.enabled:
            return
        for idx, label in enumerate(labels):
            target = self.multipliers.get(label, 1.0)
            previous = self.applied.get(label, 1.0)
            if math.isclose(target, previous, rel_tol=1e-6, abs_tol=1e-6):
                continue
            base = weights[idx] / previous if previous not in (0.0,) else weights[idx]
            weights[idx] = base * target
        if dataset_examples is not None and idx_to_label is not None:
            for example in dataset_examples:
                label_idx = int(example.label)
                if 0 <= label_idx < len(idx_to_label):
                    label = idx_to_label[label_idx]
                    target = self.multipliers.get(label, 1.0)
                    previous = self.applied.get(label, 1.0)
                    if math.isclose(target, previous, rel_tol=1e-6, abs_tol=1e-6):
                        continue
                    weight_value = float(example.weight)
                    base = weight_value / previous if previous not in (0.0,) else weight_value
                    example.weight = base * target
        for label, value in self.multipliers.items():
            self.applied[label] = value

    def stats(self) -> dict[str, Any]:
        if not self.multipliers:
            return {
                "class_balance_min": 1.0,
                "class_balance_max": 1.0,
                "class_balance_mean": 1.0,
            }
        values = list(self.multipliers.values())
        return {
            "class_balance_min": float(min(values)),
            "class_balance_max": float(max(values)),
            "class_balance_mean": float(sum(values) / len(values)),
        }

    def export(self) -> dict[str, Any]:
        return {
            "strategy": self.config.strategy,
            "boost": self.config.boost,
            "power": self.config.power,
            "momentum": self.config.momentum,
            "min_multiplier": self.config.min_multiplier,
            "max_multiplier": self.config.max_multiplier,
            "floor": self.config.floor,
            "min_support": self.config.min_support,
            "multipliers": dict(sorted(self.multipliers.items())),
            "label_counts": dict(sorted(self.label_counts.items())),
            "last_metrics": self.last_metrics,
        }


@dataclass
class DistillationConfig:
    alpha: float
    temperature: float


@dataclass
class RDropConfig:
    enabled: bool
    alpha: float
    passes: int = 2


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
        self.samples: dict[str, CurriculumSample] = {}
        self.history: list[dict[str, Any]] = []
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
    ) -> dict[str, Any] | None:
        if epoch < self.start_epoch:
            return None
        adjusted: list[tuple[float, float, float, str, str, float]] = []
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
            for _difficulty, mult, _prev, text, label, prob in hardest
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

    def apply(self, texts: Sequence[str], weights: list[float]) -> None:
        for idx, text in enumerate(texts):
            sample = self.samples.get(text)
            if sample is None:
                continue
            weights[idx] = sample.weight

    def export_metadata(self) -> dict[str, Any]:
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

    def export_metrics(self) -> dict[str, Any]:
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
class IntentDataset(Dataset[BatchExample]):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        vocab: dict[str, int],
        vocab_config: VocabularyConfig,
        label_to_idx: dict[str, int],
        max_len: int,
        sample_weights: Sequence[float] | None = None,
        tokenizer: Any | None = None,
        tokenizer_cache: Callable[[str], tuple[Sequence[int], Sequence[int]]] | None = None,
        embedding_model: Callable[[str], VectorLike] | None = None,
        teacher_logits: Sequence[Sequence[float] | None] | None = None,
        emotion_vectors: Sequence[Sequence[float]] | None = None,
        emotion_encoder: Callable[[str], Sequence[float]] | None = None,
        emotion_dim: int | None = None,
        keyword_vectors: Sequence[Sequence[float]] | None = None,
        pin_memory: bool = False,
        target_device: str | TorchDevice | None = None,
    ) -> None:
        self.examples: list[EncodedExample] = []
        self.vocab_config = vocab_config
        if sample_weights is None:
            sample_weights = [1.0] * len(texts)
        if len(sample_weights) != len(texts):
            raise ValueError("Sample weights must match the number of texts.")
        if teacher_logits is None:
            teacher_iter: Sequence[Sequence[float] | None] = [None] * len(texts)
        else:
            if len(teacher_logits) != len(texts):
                raise ValueError("Teacher logits must align with the provided texts.")
            teacher_iter = teacher_logits
        resolved_emotion_dim = int(emotion_dim or 0)
        include_emotion = resolved_emotion_dim > 0
        keyword_iter: Sequence[Sequence[float]] | None = keyword_vectors
        pin_requested = bool(pin_memory)
        if target_device is None:
            resolved_target_device: TorchDevice | None = None
        elif isinstance(target_device, TorchDevice):
            resolved_target_device = target_device
        else:
            resolved_target_device = torch.device(target_device)
        tensor_kwargs = {"pin_memory": True} if (pin_requested and resolved_target_device is None) else {}
        move_to_gpu = (
            resolved_target_device is not None and resolved_target_device.type != "cpu"
        )

        def _ensure_device(tensor: TorchTensor, *, dtype: TorchDType | None = None) -> TorchTensor:
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            if resolved_target_device is not None:
                tensor = tensor.to(resolved_target_device, non_blocking=move_to_gpu)
            elif pin_requested and tensor.device.type == "cpu" and not tensor.is_pinned():
                tensor = tensor.pin_memory()
            return tensor
        for idx_example, (text, label, weight, teacher_row) in enumerate(zip(texts, labels, sample_weights, teacher_iter)):
            if embedding_model is not None:
                vector = embedding_model(text)
                if isinstance(vector, torch.Tensor):
                    token_tensor = cast(TorchTensor, vector).detach()
                else:
                    token_tensor = torch.tensor(vector, dtype=torch.float32)
                token_tensor = _ensure_device(token_tensor, dtype=torch.float32)
                mask_tensor = torch.ones_like(
                    token_tensor,
                    dtype=torch.float32,
                    device=token_tensor.device,
                )
                if (
                    not move_to_gpu
                    and pin_requested
                    and mask_tensor.device.type == "cpu"
                    and not mask_tensor.is_pinned()
                ):
                    mask_tensor = mask_tensor.pin_memory()
            elif tokenizer_cache is not None:
                cached_ids, cached_mask = tokenizer_cache(text)
                token_tensor = torch.tensor(cached_ids, dtype=torch.long, **tensor_kwargs)
                mask_tensor = torch.tensor(cached_mask, dtype=torch.long, **tensor_kwargs)
            elif tokenizer is not None:
                encoded = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_attention_mask=True,
                )
                token_tensor = torch.tensor(encoded["input_ids"], dtype=torch.long, **tensor_kwargs)
                mask_tensor = torch.tensor(encoded["attention_mask"], dtype=torch.long, **tensor_kwargs)
            else:
                encoded = encode_text(text, vocab, max_len, config=self.vocab_config)
                token_tensor = torch.tensor(encoded, dtype=torch.long, **tensor_kwargs)
                pad_idx = vocab.get(PAD_TOKEN, 0)
                mask_tensor = (token_tensor != pad_idx).long()
                if pin_requested:
                    mask_tensor = mask_tensor.pin_memory()
            label_id = label_to_idx[label]
            if teacher_row is not None:
                teacher_tensor = torch.tensor(list(teacher_row), dtype=torch.float32)
                teacher_tensor = _ensure_device(teacher_tensor)
            else:
                if resolved_target_device is not None:
                    teacher_tensor = torch.empty(
                        0, dtype=torch.float32, device=resolved_target_device
                    )
                else:
                    teacher_tensor = torch.empty(0, dtype=torch.float32)
                    if pin_requested and not teacher_tensor.is_pinned():
                        teacher_tensor = teacher_tensor.pin_memory()
            raw_emotion: Sequence[float] | None = None
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
                emotion_tensor = _ensure_device(emotion_tensor)
            elif include_emotion:
                if resolved_target_device is not None:
                    emotion_tensor = torch.zeros(
                        resolved_emotion_dim,
                        dtype=torch.float32,
                        device=resolved_target_device,
                    )
                else:
                    emotion_tensor = torch.zeros(resolved_emotion_dim, dtype=torch.float32)
                    if pin_requested and not emotion_tensor.is_pinned():
                        emotion_tensor = emotion_tensor.pin_memory()
            else:
                if resolved_target_device is not None:
                    emotion_tensor = torch.empty(
                        0, dtype=torch.float32, device=resolved_target_device
                    )
                else:
                    emotion_tensor = torch.empty(0, dtype=torch.float32)
                    if pin_requested and not emotion_tensor.is_pinned():
                        emotion_tensor = emotion_tensor.pin_memory()
            if resolved_target_device is not None:
                keyword_tensor = torch.empty(
                    0, dtype=torch.float32, device=resolved_target_device
                )
            else:
                keyword_tensor = torch.empty(0, dtype=torch.float32)
                if pin_requested and not keyword_tensor.is_pinned():
                    keyword_tensor = keyword_tensor.pin_memory()
            if keyword_iter is not None and idx_example < len(keyword_iter):
                keyword_values = list(keyword_iter[idx_example])
                if keyword_values:
                    keyword_tensor = torch.tensor(keyword_values, dtype=torch.float32)
                    keyword_tensor = _ensure_device(keyword_tensor)
            self.examples.append(
                EncodedExample(
                    tokens=token_tensor,
                    attention_mask=mask_tensor,
                    label=label_id,
                    weight=float(weight),
                    teacher_logits=teacher_tensor,
                    emotion_vector=emotion_tensor,
                    keyword_logits=keyword_tensor,
                )
            )
        self.include_emotion = include_emotion
        self.emotion_dim = resolved_emotion_dim if include_emotion else 0
        self.include_keywords = any(example.keyword_logits.numel() > 0 for example in self.examples)
        self.keyword_dim = 0
        if self.include_keywords:
            for example in self.examples:
                if example.keyword_logits.numel() > 0:
                    self.keyword_dim = int(example.keyword_logits.numel())
                    break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> BatchExample:
        example = self.examples[index]
        items: list[TorchTensor] = [
            example.tokens,
            torch.tensor(example.label, dtype=torch.long),
            torch.tensor(example.weight, dtype=torch.float32),
            example.attention_mask,
            example.teacher_logits,
        ]
        if self.include_emotion:
            items.append(example.emotion_vector)
        if self.include_keywords:
            items.append(example.keyword_logits)
        return tuple(items)


class IntentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        num_layers: int = 1,
        attention_heads: int = 8,
        ffn_dim: int = 768,
        *,
        use_conv_head: bool = True,
        conv_kernel_sizes: Sequence[int] | None = None,
        conv_channels: int = 256,
        conv_dropout: float = 0.2,
    ) -> None:
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
            nn.Linear(ffn_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim * 2),
        )
        kernel_candidates: list[int] = []
        if conv_kernel_sizes:
            for size in conv_kernel_sizes:
                value = int(abs(size))
                if value > 0:
                    kernel_candidates.append(value)

        self.use_conv_head = bool(use_conv_head and kernel_candidates)
        self.conv_kernel_sizes = kernel_candidates if self.use_conv_head else []
        self.conv_channels = int(max(conv_channels, 1))
        self.conv_dropout = float(max(min(conv_dropout, 0.999), 0.0))
        self.conv_blocks: nn.ModuleList | None
        self.conv_norm: nn.LayerNorm | None
        if self.use_conv_head:
            self.conv_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(hidden_dim * 2, self.conv_channels, kernel_size=kernel, padding=kernel // 2),
                        nn.GELU(),
                        nn.Dropout(self.conv_dropout),
                    )
                    for kernel in self.conv_kernel_sizes
                ]
            )
            self.conv_norm = nn.LayerNorm(self.conv_channels * len(self.conv_kernel_sizes))
        else:
            self.conv_blocks = None
            self.conv_norm = None

        self.representation_dim = hidden_dim * 6
        if self.use_conv_head:
            self.representation_dim += self.conv_channels * len(self.conv_kernel_sizes)

        projection_dim_1 = max(self.representation_dim // 2, hidden_dim * 4, self.conv_channels * 2)
        projection_dim_2 = max(projection_dim_1 // 2, hidden_dim * 2, 256)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.representation_dim),
            nn.Linear(self.representation_dim, projection_dim_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim_1, projection_dim_2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim_2, num_classes),
        )

    def forward(
        self,
        inputs: TorchTensor,
        attention_mask: TorchTensor | None = None,
        *,
        return_features: bool = False,
    ) -> TorchTensor | tuple[TorchTensor, TorchTensor]:
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
            pooled_max, _ = masked_ffn.max(dim=1)
            pooled_max[torch.isinf(pooled_max)] = 0.0
        else:
            pooled_mean = ffn_output.mean(dim=1)
            pooled_max, _ = ffn_output.max(dim=1)
        features = [last_hidden, pooled_mean, pooled_max]
        if self.conv_blocks is not None and self.conv_kernel_sizes:
            conv_input = ffn_output.transpose(1, 2)
            conv_summaries: list[TorchTensor] = []
            for block in self.conv_blocks:
                conv_output = block(conv_input)
                pooled, _ = conv_output.max(dim=2)
                conv_summaries.append(pooled)
            conv_concat = torch.cat(conv_summaries, dim=1)
            if self.conv_norm is not None:
                conv_concat = self.conv_norm(conv_concat)
            features.append(conv_concat)

        representation = torch.cat(features, dim=1)
        logits = self.classifier(representation)
        if return_features:
            return logits, representation
        return logits


class TransformerIntentModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int) -> None:
        super().__init__()
        try:
            transformers_module = importlib.import_module("transformers")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'transformers' package is required for the transformer encoder. "
                "Install it via 'pip install transformers'."
            ) from exc
        AutoModelForSequenceClassification = getattr(
            transformers_module, "AutoModelForSequenceClassification"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:  # pragma: no cover - optional backend support
                pass
        if hasattr(self.model, "config") and getattr(self.model.config, "use_cache", True):
            try:
                self.model.config.use_cache = False
            except Exception:  # pragma: no cover - optional config attribute
                pass

    def forward(
        self,
        inputs: TorchTensor,
        attention_mask: TorchTensor | None = None,
        *,
        return_features: bool = False,
    ) -> TorchTensor | tuple[TorchTensor, TorchTensor]:
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


_ST_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class SoftMixtureOfExperts(nn.Module):
    """Lightweight soft mixture-of-experts block for sentence embeddings."""

    def __init__(
        self,
        input_dim: int,
        *,
        num_experts: int,
        expert_hidden_dim: int,
        activation_factory: Callable[[], nn.Module],
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        temperature: float = 1.0,
        topk: int = 0,
        utilisation_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("SoftMixtureOfExperts requires at least two experts.")
        if expert_hidden_dim < 1:
            raise ValueError("expert_hidden_dim must be positive for SoftMixtureOfExperts")
        if input_dim < 1:
            raise ValueError("input_dim must be positive for SoftMixtureOfExperts")
        self.num_experts = int(num_experts)
        self.expert_hidden_dim = int(expert_hidden_dim)
        self.temperature = float(max(1e-4, temperature))
        self.topk = int(max(0, min(topk, self.num_experts)))
        self.utilisation_momentum = float(min(max(utilisation_momentum, 0.0), 0.999))
        self.gate = nn.Linear(input_dim, self.num_experts)
        self.experts: nn.ModuleList = nn.ModuleList()
        for _ in range(self.num_experts):
            layers: list[nn.Module] = [nn.Linear(input_dim, self.expert_hidden_dim)]
            if use_layer_norm:
                layers.append(nn.LayerNorm(self.expert_hidden_dim))
            layers.append(activation_factory())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(self.expert_hidden_dim, input_dim))
            self.experts.append(nn.Sequential(*layers))
        utilisation_state = torch.zeros(self.num_experts, dtype=torch.float32)
        self.register_buffer("utilisation_state", utilisation_state)
        self.utilisation_state = cast(TorchTensor, utilisation_state)
        utilisation_batches = torch.zeros(1, dtype=torch.float32)
        self.register_buffer("utilisation_batches", utilisation_batches)
        self.utilisation_batches = cast(TorchTensor, utilisation_batches)
        self._cached_gates: TorchTensor | None = None

    def forward(self, inputs: TorchTensor) -> TorchTensor:
        gate_logits = self.gate(inputs)
        if self.topk > 0 and self.topk < self.num_experts:
            top_values, top_indices = gate_logits.topk(self.topk, dim=-1)
            mask = torch.full_like(gate_logits, float("-inf"))
            mask.scatter_(1, top_indices, top_values)
            gate_logits = mask
        gates = F.softmax(gate_logits / self.temperature, dim=-1)
        expert_outputs = torch.stack([expert(inputs) for expert in self.experts], dim=1)
        combined = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
        self._cached_gates = gates
        if self.training:
            with torch.no_grad():
                utilisation = gates.mean(dim=0)
                self.utilisation_state.mul_(self.utilisation_momentum).add_(
                    utilisation * (1.0 - self.utilisation_momentum)
                )
                self.utilisation_batches += 1.0
        return combined

    def compute_regulariser(
        self,
        entropy_weight: float,
        balance_weight: float,
    ) -> tuple[TorchTensor, dict[str, Any]]:
        gates = self._cached_gates
        if gates is None:
            zero = torch.zeros(0, device=self.gate.weight.device, dtype=self.gate.weight.dtype)
            return zero, {
                "kind": "moe",
                "loss": 0.0,
                "entropy": 0.0,
                "entropy_gap": 0.0,
                "balance": 0.0,
                "max_gate": 0.0,
                "active": 0.0,
                "samples": 0.0,
                "utilisation_mean": float(self.utilisation_state.mean().item()) if self.utilisation_state.numel() else 0.0,
                "utilisation_min": float(self.utilisation_state.min().item()) if self.utilisation_state.numel() else 0.0,
                "utilisation_max": float(self.utilisation_state.max().item()) if self.utilisation_state.numel() else 0.0,
            }
        eps = 1e-8
        batch = gates.shape[0]
        loss = torch.zeros(batch, device=gates.device, dtype=gates.dtype)
        entropy = -(gates * (gates + eps).log()).sum(dim=-1)
        max_entropy = math.log(max(self.num_experts, 1))
        entropy_gap = (max_entropy - entropy).clamp_min(0.0)
        if entropy_weight > 0:
            loss = loss + entropy_gap * entropy_weight
        mean_gate = gates.mean(dim=0)
        uniform = torch.full_like(mean_gate, 1.0 / max(self.num_experts, 1))
        balance_metric = F.mse_loss(mean_gate, uniform, reduction="sum") / max(self.num_experts, 1)
        if balance_weight > 0:
            loss = loss + balance_metric * balance_weight
        max_gate, _ = gates.max(dim=-1)
        threshold = max(0.05, 1.0 / max(self.num_experts, 1))
        active = (gates > threshold).float().sum(dim=-1)
        summary = {
            "kind": "moe",
            "loss": float(loss.detach().mean().item()),
            "entropy": float(entropy.detach().mean().item()),
            "entropy_gap": float(entropy_gap.detach().mean().item()),
            "balance": float(balance_metric.detach().item()),
            "max_gate": float(max_gate.detach().mean().item()),
            "active": float(active.detach().mean().item()),
            "samples": float(batch),
            "utilisation_mean": float(self.utilisation_state.detach().mean().item())
            if self.utilisation_state.numel()
            else 0.0,
            "utilisation_min": float(self.utilisation_state.detach().min().item())
            if self.utilisation_state.numel()
            else 0.0,
            "utilisation_max": float(self.utilisation_state.detach().max().item())
            if self.utilisation_state.numel()
            else 0.0,
        }
        self._cached_gates = None
        return loss, summary

class SentenceTransformerClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        *,
        num_layers: int = 1,
        expansion: float = 1.0,
        activation: str = "relu",
        final_dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_residual: bool = False,
        moe_experts: int = 0,
        moe_hidden_dim: int = 0,
        moe_activation: str | None = None,
        moe_dropout: float = 0.0,
        moe_temperature: float = 1.0,
        moe_topk: int = 0,
        moe_entropy_weight: float = 0.0,
        moe_balance_weight: float = 0.0,
        moe_use_layer_norm: bool = False,
        moe_utilisation_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        activation_key = activation.lower()
        if activation_key not in _ST_ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{activation}'. Choose from {sorted(_ST_ACTIVATIONS)}.")
        self.activation_name = activation_key
        activation_factory = _ST_ACTIVATIONS[activation_key]
        hidden_dims = progressive_mlp_hidden_dims(hidden_dim, num_layers, expansion)
        self.hidden_dims = hidden_dims
        self.blocks = nn.ModuleList()
        input_dim = embedding_dim
        for width in hidden_dims:
            layers: list[nn.Module] = [nn.Linear(input_dim, width)]
            if use_layer_norm:
                layers.append(nn.LayerNorm(width))
            layers.append(activation_factory())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.blocks.append(nn.Sequential(*layers))
            input_dim = width
        self.moe_entropy_weight = max(0.0, float(moe_entropy_weight))
        self.moe_balance_weight = max(0.0, float(moe_balance_weight))
        moe_expert_count = int(max(0, moe_experts))
        self.moe_layer: SoftMixtureOfExperts | None
        if moe_expert_count >= 2:
            moe_activation_key = (moe_activation or activation).lower()
            if moe_activation_key not in _ST_ACTIVATIONS:
                raise ValueError(
                    f"Unsupported MoE activation '{moe_activation}'. Choose from {sorted(_ST_ACTIVATIONS)}."
                )
            moe_activation_factory = _ST_ACTIVATIONS[moe_activation_key]
            expert_hidden = int(max(moe_hidden_dim, input_dim)) if moe_hidden_dim > 0 else int(max(input_dim, hidden_dim))
            self.moe_layer = SoftMixtureOfExperts(
                input_dim,
                num_experts=moe_expert_count,
                expert_hidden_dim=expert_hidden,
                activation_factory=moe_activation_factory,
                dropout=max(0.0, moe_dropout),
                use_layer_norm=bool(moe_use_layer_norm),
                temperature=float(moe_temperature),
                topk=int(max(0, moe_topk)),
                utilisation_momentum=float(moe_utilisation_momentum),
            )
        else:
            self.moe_layer = None
        self.final_dropout_layer = nn.Dropout(final_dropout) if final_dropout > 0 else None
        self.output_layer = nn.Linear(input_dim, num_classes)
        self.use_residual = bool(use_residual)
        self._latest_moe_summary: dict[str, Any] = {}

    def forward(
        self,
        inputs: TorchTensor,
        attention_mask: TorchTensor | None = None,
        *,
        return_features: bool = False,
    ) -> TorchTensor | tuple[TorchTensor, TorchTensor]:
        hidden = inputs
        for block in self.blocks:
            residual = hidden
            hidden = block(hidden)
            if self.use_residual and hidden.shape == residual.shape:
                hidden = hidden + residual
        if self.moe_layer is not None:
            hidden = self.moe_layer(hidden)
        representation = hidden
        if self.final_dropout_layer is not None:
            representation = self.final_dropout_layer(representation)
        logits = self.output_layer(representation)
        if return_features:
            return logits, representation
        return logits

    def compute_extra_losses(self) -> tuple[TorchTensor | None, dict[str, Any] | None]:
        if self.moe_layer is None:
            self._latest_moe_summary = {}
            return None, None
        losses, summary = self.moe_layer.compute_regulariser(
            self.moe_entropy_weight,
            self.moe_balance_weight,
        )
        self._latest_moe_summary = summary
        if losses.numel() == 0:
            return None, summary
        return losses, summary

    def export_moe_state(self) -> dict[str, Any] | None:
        if self.moe_layer is None or not _torch_available or torch is None:
            return None
        utilisation_state = cast(TorchTensor | None, getattr(self.moe_layer, "utilisation_state", None))
        if isinstance(utilisation_state, TorchTensorType):
            utilisation = utilisation_state.detach().cpu().tolist()
        else:
            utilisation = []
        utilisation_batches = cast(TorchTensor | None, getattr(self.moe_layer, "utilisation_batches", None))
        if isinstance(utilisation_batches, TorchTensorType):
            batches = float(utilisation_batches.detach().cpu().item())
        elif utilisation_batches is not None:
            batches = float(cast(Any, utilisation_batches))
        else:
            batches = 0.0
        return {
            "num_experts": int(self.moe_layer.num_experts),
            "expert_hidden_dim": int(self.moe_layer.expert_hidden_dim),
            "utilisation": utilisation,
            "utilisation_batches": batches,
            "entropy_weight": float(self.moe_entropy_weight),
            "balance_weight": float(self.moe_balance_weight),
        }


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
        inputs: TorchTensor,
        attention_mask: TorchTensor | None = None,
        *,
        emotion_features: TorchTensor | None = None,
        return_components: bool = False,
        return_features: bool = False,
    ) -> (
        TorchTensor |
        tuple[TorchTensor, TorchTensor] |
        tuple[TorchTensor, TorchTensor, TorchTensor] |
        tuple[TorchTensor, TorchTensor, TorchTensor, TorchTensor]
    ):
        base_features: TorchTensor | None = None
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
            emotion_tensor = emotion_features
            if emotion_tensor.dim() == 1:
                emotion_tensor = emotion_tensor.unsqueeze(0)
            emotion_tensor = emotion_tensor.to(base_logits.dtype)
            if emotion_tensor.shape[-1] == self.num_emotions:
                emotion_logits = self.emotion_transform(emotion_tensor)
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

    def compute_extra_losses(self) -> tuple[TorchTensor | None, dict[str, Any] | None]:
        base_method = getattr(self.base_model, "compute_extra_losses", None)
        if callable(base_method):
            result = base_method()
            if isinstance(result, tuple):
                sequence_tuple = cast(tuple[object, ...], result)
                loss_part: object | None = sequence_tuple[0] if sequence_tuple else None
                summary_part: object | None = (
                    sequence_tuple[1] if len(sequence_tuple) > 1 else None
                )
                loss_tensor: TorchTensor | None
                if isinstance(loss_part, TorchTensorType):
                    loss_tensor = loss_part
                elif loss_part is None:
                    loss_tensor = None
                else:
                    loss_tensor = torch.as_tensor(loss_part)
                summary_dict: dict[str, Any] | None = None
                if isinstance(summary_part, Mapping):
                    summary_mapping = cast(Mapping[Any, Any], summary_part)
                    summary_dict = _stringify_mapping(summary_mapping)
                return loss_tensor, summary_dict
            if isinstance(result, TorchTensorType):
                return result, None
            if result is None:
                return None, None
            if isinstance(result, (int, float)):
                try:
                    base_param = next(iter(self.base_model.parameters()))
                    device = base_param.device
                except StopIteration:
                    device = torch.device("cpu")
                return torch.tensor(float(result), device=device), None
            return torch.as_tensor(result), None
        return None, None

    def export_moe_state(self) -> dict[str, Any] | None:
        base_method = getattr(self.base_model, "export_moe_state", None)
        if callable(base_method):
            result = base_method()
            if isinstance(result, Mapping):
                return _stringify_mapping(cast(Mapping[Any, Any], result))
            return None
        return None


def _micro_batch_slices(batch_size: int, micro_batch: int) -> Iterable[tuple[int, int]]:
    if batch_size <= 0:
        return [(0, 0)]
    if micro_batch <= 0 or micro_batch >= batch_size:
        return [(0, batch_size)]
    slices: list[tuple[int, int]] = []
    for start in range(0, batch_size, micro_batch):
        end = min(start + micro_batch, batch_size)
        slices.append((start, end))
    return slices


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader[BatchExample],
    criterion: nn.Module,
    optimizer: TorchOptimizer,
    device: TorchDevice,
    *,
    scheduler: SchedulerLike | None = None,
    scheduler_step_per_batch: bool = False,
    scaler: _GradScalerProtocol | None = None,
    amp_enabled: bool = False,
    amp_device_type: str = "cuda",
    max_grad_norm: float | None = None,
    distillation_config: DistillationConfig | None = None,
    emotion_config: EmotionTrainingConfig | None = None,
    meta_config: MetaCognitiveConfig | None = None,
    neuro_config: NeuroSymbolicConfig | None = None,
    discovery_config: SelfDiscoveryConfig | None = None,
    transcendent_config: TranscendentCognitionConfig | None = None,
    frontier_config: FrontierIntelligenceConfig | None = None,
    rdrop_config: RDropConfig | None = None,
    collect_performance_stats: bool = False,
    memory_regulator: VramBudgetRegulator | None = None,
    micro_batch_size: int | None = None,
    memory_sweeper: CudaMemorySweeper | None = None,
) -> tuple[float, float, dict[str, Any]]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    non_blocking = device.type in {"cuda", "mps"}
    amp_enabled_flag = bool(amp_enabled)
    dataset_obj = getattr(dataloader, "dataset", None)
    dataset_has_emotion = bool(getattr(dataset_obj, "include_emotion", False))
    dataset_has_keywords = bool(getattr(dataset_obj, "include_keywords", False))

    grad_accumulation_steps = max(1, getattr(optimizer, "grad_accumulation_steps", 1))
    ema_model: nn.Module | None = getattr(optimizer, "ema_model", None)
    ema_active: bool = bool(getattr(optimizer, "ema_active", False))
    swa_model: AveragedModel | None = getattr(optimizer, "swa_model", None)
    swa_active: bool = bool(getattr(optimizer, "swa_active", False))

    optimizer_steps = 0
    ema_updates = 0
    swa_updates = 0

    optimizer.zero_grad(set_to_none=True)
    num_batches = len(dataloader)
    epoch_start_time = time.perf_counter()
    token_count_total = 0.0
    batch_counter = 0
    device_is_cuda = device.type == "cuda" and torch.cuda.is_available()
    if device_is_cuda:
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
    memory_sweep_calls = 0
    memory_sweep_duration = 0.0
    memory_sweep_last_free = float("nan")
    memory_sweep_last_reserved = float("nan")
    dataloader_batch_size = int(getattr(dataloader, "batch_size", 0) or 0)
    micro_hint_initial = micro_batch_size or getattr(optimizer, "micro_batch_size", 0) or dataloader_batch_size
    if not micro_hint_initial or micro_hint_initial <= 0:
        micro_hint_initial = max(1, dataloader_batch_size or 1)
    optimizer.micro_batch_size = int(max(1, micro_hint_initial))
    micro_step_counter = 0
    if num_batches == 0:
        return 0.0, 0.0, {
            "optimizer_steps": 0,
            "ema_updates": 0,
            "swa_updates": 0,
            "duration": 0.0,
            "examples": 0.0,
            "tokens": 0.0,
            "batches": 0.0,
        }

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

    moe_loss_total = 0.0
    moe_entropy_total = 0.0
    moe_gap_total = 0.0
    moe_balance_total = 0.0
    moe_active_total = 0.0
    moe_max_total = 0.0
    moe_sample_total = 0.0
    moe_batches = 0
    moe_util_mean_total = 0.0
    moe_util_min_total = 0.0
    moe_util_max_total = 0.0

    rdrop_enabled = (
        rdrop_config is not None
        and rdrop_config.enabled
        and rdrop_config.alpha > 0.0
        and rdrop_config.passes >= 2
    )
    rdrop_alpha = float(rdrop_config.alpha) if rdrop_enabled and rdrop_config else 0.0
    rdrop_passes = int(rdrop_config.passes) if rdrop_enabled and rdrop_config else 0
    rdrop_kl_total = 0.0
    rdrop_loss_total = 0.0
    rdrop_batches = 0
    regulator = memory_regulator


    for batch_idx, batch in enumerate(dataloader, start=1):
        batch_counter += 1
        if len(batch) < 5:
            raise ValueError("Batches must contain at least tokens, labels, weights, attention mask, and teacher logits.")
        inputs_raw, targets_raw, weights_raw, attention_candidate = batch[:4]
        batch_inputs = torch.as_tensor(inputs_raw)
        batch_targets = torch.as_tensor(targets_raw)
        batch_weights = torch.as_tensor(weights_raw)
        batch_attention: TorchTensor | None = None
        if attention_candidate is not None:
            if torch.is_tensor(attention_candidate):
                batch_attention = attention_candidate
            else:
                batch_attention = torch.as_tensor(attention_candidate)
        teacher_candidate = batch[4]
        try:
            batch_teacher = torch.as_tensor(teacher_candidate)
        except Exception:
            batch_teacher = None
        next_index = 5
        batch_emotion: TorchTensor | None = None
        if dataset_has_emotion and len(batch) > next_index:
            emotion_candidate = batch[next_index]
            batch_emotion = torch.as_tensor(emotion_candidate)
            next_index += 1
        batch_keywords: TorchTensor | None = None
        if dataset_has_keywords and len(batch) > next_index:
            keyword_candidate = batch[next_index]
            batch_keywords = torch.as_tensor(keyword_candidate)

        if collect_performance_stats:
            if batch_attention is not None:
                try:
                    token_count_total += float(batch_attention.sum().item())
                except Exception:
                    try:
                        token_count_total += float(np.sum(batch_attention))  # type: ignore[arg-type]
                    except Exception:
                        pass
        batch_size_current = batch_inputs.size(0)

        micro_hint = int(max(1, getattr(optimizer, "micro_batch_size", optimizer.micro_batch_size)))
        effective_micro = min(batch_size_current, micro_hint)
        if regulator is not None:
            adjusted_micro = regulator.micro_batch_size_for(batch_size_current, effective_micro)
            effective_micro = int(max(1, adjusted_micro))
            optimizer.micro_batch_size = max(1, effective_micro)
        slices = list(_micro_batch_slices(batch_size_current, effective_micro))
        if not slices:
            slices = [(0, batch_size_current)]
        slice_count = len(slices)
        try:
            global_weight_value = float(batch_weights.detach().sum().item())
        except Exception:
            global_weight_value = float(batch_size_current)
        if not math.isfinite(global_weight_value) or global_weight_value <= 0.0:
            global_weight_value = float(batch_size_current)
        global_weight_denominator = torch.tensor(
            global_weight_value,
            device=device,
            dtype=torch.float32,
        )

        for slice_position, (slice_start, slice_end) in enumerate(slices, start=1):
            input_slice = batch_inputs[slice_start:slice_end].to(device, non_blocking=non_blocking)
            target_slice = batch_targets[slice_start:slice_end].to(device, non_blocking=non_blocking)
            weight_slice = batch_weights[slice_start:slice_end].to(device, non_blocking=non_blocking)
            attention_slice: TorchTensor | None = None
            if batch_attention is not None:
                base_attention = batch_attention.to(device, non_blocking=non_blocking)
                if (
                    base_attention.dim() >= 1
                    and base_attention.size(0) == batch_size_current
                ):
                    attention_slice = base_attention[slice_start:slice_end]
                else:
                    attention_slice = base_attention

            if attention_slice is not None and attention_slice.size(0) != input_slice.size(0):
                raise ValueError("Attention mask size does not match input batch size")

            inputs = input_slice
            targets = target_slice
            weights = weight_slice
            attention_mask = attention_slice

            teacher_slice: TorchTensor | None = None
            if batch_teacher is not None and batch_teacher.numel() > 0:
                teacher_candidate = batch_teacher
                if teacher_candidate.dim() >= 1 and teacher_candidate.size(0) == batch_size_current:
                    teacher_candidate = teacher_candidate[slice_start:slice_end]
                teacher_slice = teacher_candidate.to(device, non_blocking=non_blocking)

            emotion_slice: TorchTensor | None = None
            if batch_emotion is not None and batch_emotion.numel() > 0:
                emotion_candidate = batch_emotion
                if emotion_candidate.dim() >= 2 and emotion_candidate.size(0) == batch_size_current:
                    emotion_candidate = emotion_candidate[slice_start:slice_end]
                elif emotion_candidate.dim() == 1 and emotion_candidate.size(0) == batch_size_current:
                    emotion_candidate = emotion_candidate[slice_start:slice_end].unsqueeze(0)
                processed_emotion = emotion_candidate.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=non_blocking,
                )
                if processed_emotion.dim() > 0 and processed_emotion.numel() > 0:
                    if processed_emotion.dim() == 1:
                        processed_emotion = processed_emotion.unsqueeze(0)
                    emotion_slice = processed_emotion

            keyword_slice: TorchTensor | None = None
            if batch_keywords is not None and batch_keywords.numel() > 0:
                keyword_candidate = batch_keywords
                if keyword_candidate.dim() >= 2 and keyword_candidate.size(0) == batch_size_current:
                    keyword_candidate = keyword_candidate[slice_start:slice_end]
                elif keyword_candidate.dim() == 1 and keyword_candidate.size(0) == batch_size_current:
                    keyword_candidate = keyword_candidate[slice_start:slice_end].unsqueeze(0)
                processed_keyword = keyword_candidate.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=non_blocking,
                )
                if processed_keyword.dim() > 0 and processed_keyword.numel() > 0:
                    if processed_keyword.dim() == 1:
                        processed_keyword = processed_keyword.unsqueeze(0)
                    keyword_slice = processed_keyword

            supports_emotion = (
                emotion_slice is not None
                and emotion_slice.numel() > 0
                and emotion_config is not None
                and emotion_config.enabled
                and getattr(model, "supports_emotion_features", False)
            )

            context = autocast_context(amp_enabled_flag, amp_device_type)

            with context:
                features: TorchTensor | None = None
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
                            emotion_features=emotion_slice,
                            return_components=True,
                            return_features=True,
                        )
                        logits, _, _, features = outputs
                    else:
                        logits, _, _ = model(
                            inputs,
                            attention_mask=attention_mask,
                            emotion_features=emotion_slice,
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
                if keyword_slice is not None and keyword_slice.shape[-1] == logits.shape[-1]:
                    logits = logits + keyword_slice
                rdrop_logits: list[TorchTensor] = []
                if rdrop_enabled:
                    for _ in range(max(0, rdrop_passes - 1)):
                        if supports_emotion:
                            alt = model(
                                inputs,
                                attention_mask=attention_mask,
                                emotion_features=emotion_slice,
                            )
                        else:
                            alt = model(inputs, attention_mask=attention_mask)
                        rdrop_logits.append(alt if isinstance(alt, TorchTensor) else alt[0])
                    if keyword_slice is not None and keyword_slice.shape[-1] == logits.shape[-1]:
                        for idx_alt, alt_logits in enumerate(rdrop_logits):
                            rdrop_logits[idx_alt] = alt_logits + keyword_slice
                hard_loss = criterion(logits, targets)
                if hard_loss.dim() == 0:
                    hard_loss = hard_loss.unsqueeze(0)
                if (
                    distillation_config is not None
                    and teacher_slice is not None
                    and teacher_slice.numel() > 0
                ):
                    kd_loss = compute_distillation_loss(
                        logits,
                        teacher_slice,
                        temperature=distillation_config.temperature,
                    )
                    loss_values = (
                        hard_loss * (1.0 - distillation_config.alpha)
                        + kd_loss * distillation_config.alpha
                    )
                else:
                    loss_values = hard_loss
                rdrop_component: TorchTensor | None = None
                if (
                    supports_emotion
                    and emotion_slice is not None
                    and emotion_config is not None
                ):
                    alignment = emotion_config.memory.alignment_loss(
                        logits,
                        emotion_slice,
                        temperature=emotion_config.temperature,
                    )
                    loss_values = loss_values + alignment * emotion_config.weight
                    emotion_alignment_total += float(alignment.detach().mean().item())
                    emotion_batches += 1
                meta_summary: dict[str, Any] | None = None
                neuro_summary: dict[str, Any] | None = None
                discovery_summary: dict[str, Any] | None = None
                transcendent_summary: dict[str, Any] | None = None
                frontier_summary: dict[str, Any] | None = None
                extra_summary: dict[str, Any] | None = None
                extra_losses: TorchTensor | None = None
                compute_extra = getattr(model, "compute_extra_losses", None)
                if callable(compute_extra):
                    result = compute_extra()
                    if isinstance(result, tuple):
                        result_tuple = cast(tuple[object, ...], result)
                        loss_part: object | None = (
                            result_tuple[0] if result_tuple else None
                        )
                        summary_part: object | None = (
                            result_tuple[1] if len(result_tuple) > 1 else None
                        )
                        if isinstance(loss_part, TorchTensorType):
                            extra_losses = loss_part
                        elif loss_part is None:
                            extra_losses = None
                        else:
                            extra_losses = torch.as_tensor(loss_part)
                        if isinstance(summary_part, Mapping):
                            summary_mapping = cast(Mapping[Any, Any], summary_part)
                            extra_summary = _stringify_mapping(summary_mapping)
                        else:
                            extra_summary = None
                    else:
                        if isinstance(result, TorchTensorType):
                            extra_losses = result
                        elif result is None:
                            extra_losses = None
                        else:
                            extra_losses = torch.as_tensor(result)
                if (
                    meta_config is not None
                    and meta_config.enabled
                    and features is not None
                ):
                    regulariser, meta_summary = meta_config.introspector.compute_regulariser(
                        features,
                        targets,
                        logits,
                        meta_config,
                    )
                    loss_values = loss_values + regulariser
                if (
                    neuro_config is not None
                    and neuro_config.enabled
                    and features is not None
                ):
                    ns_loss, neuro_summary = neuro_config.reasoner.compute_loss(
                        features,
                        logits,
                        targets,
                        emotion_features=emotion_slice,
                        config=neuro_config,
                    )
                    loss_values = loss_values + ns_loss
                if (
                    discovery_config is not None
                    and discovery_config.enabled
                    and features is not None
                ):
                    discovery_loss, discovery_summary = discovery_config.orchestrator.compute_loss(
                        features,
                        logits,
                        targets,
                        config=discovery_config,
                        emotion_features=emotion_slice,
                    )
                    loss_values = loss_values + discovery_loss
                if (
                    transcendent_config is not None
                    and transcendent_config.enabled
                    and features is not None
                ):
                    transcendent_loss, transcendent_summary = transcendent_config.architect.compute_loss(
                        features,
                        logits,
                        targets,
                        transcendent_config,
                        emotion_features=emotion_slice,
                    )
                    loss_values = loss_values + transcendent_loss
                if (
                    frontier_config is not None
                    and frontier_config.enabled
                    and features is not None
                ):
                    frontier_loss, frontier_summary = frontier_config.catalyst.compute_loss(
                        features,
                        logits,
                        targets,
                        frontier_config,
                        emotion_features=emotion_slice,
                    )
                    loss_values = loss_values + frontier_loss
                if extra_losses is not None and extra_losses.numel() > 0:
                    if extra_losses.dim() == 0:
                        extra_losses = extra_losses.unsqueeze(0)
                    loss_values = loss_values + extra_losses
                if rdrop_enabled and rdrop_logits:
                    sym_kl: TorchTensor | None = None
                    for alt_logits in rdrop_logits:
                        current = symmetric_kl_divergence(logits, alt_logits)
                        sym_kl = current if sym_kl is None else sym_kl + current
                    assert sym_kl is not None
                    component = sym_kl / max(len(rdrop_logits), 1)
                    loss_values = loss_values + component * rdrop_alpha
                    rdrop_component = component
                weight_denominator = global_weight_denominator
                if rdrop_component is not None:
                    weighted_kl = (rdrop_component * weights).sum() / weight_denominator
                    rdrop_kl_total += float(weighted_kl.detach().item())
                    rdrop_loss_total += float((weighted_kl * rdrop_alpha).detach().item())
                    rdrop_batches += 1
                weighted_loss = (loss_values * weights).sum() / weight_denominator

            raw_batch_loss = hard_loss.detach().mean().item()
            total_loss += raw_batch_loss * targets.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            loss_for_backprop = weighted_loss / grad_accumulation_steps

            if (
                meta_config is not None
                and meta_config.enabled
                and features is not None
            ):
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

            if (
                neuro_config is not None
                and neuro_config.enabled
                and features is not None
            ):
                detached_emotion: TorchTensor | None
                if emotion_slice is not None and emotion_slice.numel() > 0:
                    detached_emotion = emotion_slice.detach()
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
            if (
                discovery_config is not None
                and discovery_config.enabled
                and features is not None
            ):
                detached_emotion: TorchTensor | None
                if emotion_slice is not None and emotion_slice.numel() > 0:
                    detached_emotion = emotion_slice.detach()
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
            if (
                transcendent_config is not None
                and transcendent_config.enabled
                and features is not None
            ):
                detached_emotion: TorchTensor | None
                if emotion_slice is not None and emotion_slice.numel() > 0:
                    detached_emotion = emotion_slice.detach()
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
            if (
                frontier_config is not None
                and frontier_config.enabled
                and features is not None
            ):
                detached_emotion: TorchTensor | None
                if emotion_slice is not None and emotion_slice.numel() > 0:
                    detached_emotion = emotion_slice.detach()
                else:
                    detached_emotion = None
                frontier_config.catalyst.update_state(
                    features.detach(),
                    logits.detach(),
                    targets.detach(),
                    emotion_features=detached_emotion,
                    config=frontier_config,
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
            if extra_summary is not None and extra_summary.get("kind") == "moe":
                sample_count = float(extra_summary.get("samples", 0.0))
                if sample_count > 0:
                    moe_sample_total += sample_count
                    moe_loss_total += extra_summary.get("loss", 0.0) * sample_count
                    moe_entropy_total += extra_summary.get("entropy", 0.0) * sample_count
                    moe_gap_total += extra_summary.get("entropy_gap", 0.0) * sample_count
                    moe_balance_total += extra_summary.get("balance", 0.0) * sample_count
                    moe_active_total += extra_summary.get("active", 0.0) * sample_count
                    moe_max_total += extra_summary.get("max_gate", 0.0) * sample_count
                if "utilisation_mean" in extra_summary:
                    moe_util_mean_total += extra_summary.get("utilisation_mean", 0.0)
                    moe_util_min_total += extra_summary.get("utilisation_min", 0.0)
                    moe_util_max_total += extra_summary.get("utilisation_max", 0.0)
                if sample_count > 0 or "utilisation_mean" in extra_summary:
                    moe_batches += 1

            if amp_enabled_flag and scaler is not None:
                scaler.scale(loss_for_backprop).backward()
            else:
                loss_for_backprop.backward()

            micro_step_counter += 1
            if device_is_cuda:
                if regulator is not None:
                    observation: dict[str, Any] | None
                    try:
                        observation = regulator.observe(
                            step_index=micro_step_counter,
                            suggested=int(max(1, getattr(optimizer, "micro_batch_size", effective_micro))),
                        )
                    except Exception:
                        observation = None
                    adjustment = _snapshot_mapping(observation)
                    if adjustment is not None:
                        new_size = int(adjustment.get("micro_batch", getattr(optimizer, "micro_batch_size", 1)))
                        action = str(adjustment.get("action", "adjust"))
                        free_gb = float(adjustment.get("free_gb", float("nan")))
                        utilization = float(adjustment.get("utilization", float("nan")))
                        utilization_limit = float(adjustment.get("utilization_limit", float("nan")))
                        configured_limit = float(adjustment.get("configured_max_utilization", float("nan")))
                        optimizer.micro_batch_size = max(1, int(new_size))
                        util_display = f"{utilization:.2%}" if math.isfinite(utilization) else "n/a"
                        limit_display = f"{utilization_limit:.2%}" if math.isfinite(utilization_limit) else "n/a"
                        configured_display = (
                            f"{configured_limit:.2%}" if math.isfinite(configured_limit) else "n/a"
                        )
                        free_display = f"{free_gb:.2f}" if math.isfinite(free_gb) else "n/a"
                        print(
                            "VRAM regulator: {action} micro-batch to {size} "
                            "(utilization {util} | limit {limit} | configured max {configured} | free ≈ {free} GiB).".format(
                                action=action,
                                size=int(new_size),
                                util=util_display,
                                limit=limit_display,
                                configured=configured_display,
                                free=free_display,
                            )
                        )
                sweep_record: dict[str, Any] | None = None
                if memory_sweeper is not None:
                    try:
                        sweep_record = memory_sweeper.maybe_sweep(micro_step_counter)
                    except Exception:
                        sweep_record = None
                    if sweep_record is not None:
                        memory_sweep_calls += 1
                        memory_sweep_duration += float(sweep_record.get("duration", 0.0))
                        memory_sweep_last_reserved = float(sweep_record.get("reserved_gb", float("nan")))
                        memory_sweep_last_free = float(sweep_record.get("free_gb_after", float("nan")))
                        before_free = sweep_record.get("free_gb_before")
                        after_free = sweep_record.get("free_gb_after")
                        if after_free is not None and math.isfinite(after_free):
                            after_display = f"{after_free:.2f}"
                        else:
                            after_display = "n/a"
                        if before_free is not None and math.isfinite(before_free):
                            before_display = f"{before_free:.2f}"
                        else:
                            before_display = "n/a"
                        print(
                            "VRAM sweeper: cleared cache (reserved {reserved:.2f} GiB | free {before}→{after} GiB).".format(
                                reserved=float(sweep_record.get("reserved_gb", 0.0)),
                                before=before_display,
                                after=after_display,
                            )
                        )
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception:
                    pass

            if isinstance(inputs, torch.Tensor):
                del inputs
            if isinstance(targets, torch.Tensor):
                del targets
            if isinstance(weights, torch.Tensor):
                del weights
            if isinstance(attention_mask, torch.Tensor):
                del attention_mask
            if isinstance(emotion_slice, torch.Tensor):
                del emotion_slice
            if isinstance(keyword_slice, torch.Tensor):
                del keyword_slice
            if isinstance(teacher_slice, torch.Tensor):
                del teacher_slice
            if isinstance(logits, torch.Tensor):
                del logits
            if isinstance(features, torch.Tensor):
                del features
            if isinstance(loss_values, torch.Tensor):
                del loss_values
            if isinstance(hard_loss, torch.Tensor):
                del hard_loss

            last_micro_in_epoch = (batch_idx == num_batches) and (slice_position == slice_count)
            should_step = (micro_step_counter % grad_accumulation_steps == 0) or last_micro_in_epoch

            if should_step:
                if amp_enabled_flag and scaler is not None:
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
    epoch_duration = time.perf_counter() - epoch_start_time
    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
        {
            "optimizer_steps": optimizer_steps,
            "ema_updates": ema_updates,
            "swa_updates": swa_updates,
            "memory_sweeps": float(memory_sweep_calls),
            "memory_sweep_time": float(memory_sweep_duration),
            "memory_sweep_last_free": memory_sweep_last_free,
            "memory_sweep_last_reserved": memory_sweep_last_reserved,
            "emotion_alignment": (emotion_alignment_total / emotion_batches) if emotion_batches else 0.0,
            "meta_loss": (meta_loss_total / meta_sample_total) if meta_sample_total else 0.0,
            "meta_attraction": (meta_attr_total / meta_sample_total) if meta_sample_total else 0.0,
            "meta_repulsion": (meta_rep_total / meta_sample_total) if meta_sample_total else 0.0,
            "meta_novelty": (meta_novel_total / meta_sample_total) if meta_sample_total else 0.0,
            "meta_gap": (meta_gap_total / meta_sample_total) if meta_sample_total else 0.0,
            "meta_entropy": (meta_entropy_total / meta_sample_total) if meta_sample_total else 0.0,
            "meta_coverage": (meta_coverage_total / meta_batches) if meta_batches else 0.0,
            "meta_updates": (
                int(meta_config.introspector.total_updates)
                if meta_config is not None and meta_config.enabled
                else 0
            ),
            "neuro_loss": (neuro_loss_total / neuro_sample_total) if neuro_sample_total else 0.0,
            "neuro_structural": (neuro_struct_total / neuro_sample_total) if neuro_sample_total else 0.0,
            "neuro_semantic": (neuro_semantic_total / neuro_sample_total) if neuro_sample_total else 0.0,
            "neuro_affective": (neuro_affective_total / neuro_sample_total) if neuro_sample_total else 0.0,
            "neuro_entropy": (neuro_entropy_total / neuro_sample_total) if neuro_sample_total else 0.0,
            "neuro_cohesion": (neuro_cohesion_total / neuro_sample_total) if neuro_sample_total else 0.0,
            "neuro_updates": (
                int(neuro_config.reasoner.total_updates)
                if neuro_config is not None and neuro_config.enabled
                else 0
            ),
            "discovery_loss": (discovery_loss_total / discovery_sample_total) if discovery_sample_total else 0.0,
            "discovery_alignment": (discovery_alignment_total / discovery_sample_total) if discovery_sample_total else 0.0,
            "discovery_contrast": (discovery_contrast_total / discovery_sample_total) if discovery_sample_total else 0.0,
            "discovery_imagination": (discovery_imagination_total / discovery_sample_total) if discovery_sample_total else 0.0,
            "discovery_emotion": (discovery_emotion_total / discovery_sample_total) if discovery_sample_total else 0.0,
            "discovery_confidence": (discovery_confidence_total / discovery_sample_total) if discovery_sample_total else 0.0,
            "discovery_curiosity": (discovery_curiosity_total / discovery_sample_total) if discovery_sample_total else 0.0,
            "discovery_counter_share": (discovery_counter_share_total / discovery_batches) if discovery_batches else 0.0,
            "discovery_updates": (
                int(discovery_config.orchestrator.total_updates)
                if discovery_config is not None and discovery_config.enabled
                else 0
            ),
            "transcendent_loss": (transcendent_loss_total / transcendent_sample_total) if transcendent_sample_total else 0.0,
            "transcendent_stability": (transcendent_stability_total / transcendent_sample_total) if transcendent_sample_total else 0.0,
            "transcendent_divergence": (transcendent_divergence_total / transcendent_sample_total) if transcendent_sample_total else 0.0,
            "transcendent_foresight": (transcendent_foresight_total / transcendent_sample_total) if transcendent_sample_total else 0.0,
            "transcendent_synthesis": (transcendent_synthesis_total / transcendent_sample_total) if transcendent_sample_total else 0.0,
            "transcendent_affective": (transcendent_affective_total / transcendent_sample_total) if transcendent_sample_total else 0.0,
            "transcendent_entropy": (transcendent_entropy_total / transcendent_sample_total) if transcendent_sample_total else 0.0,
            "transcendent_coherence": (transcendent_coherence_total / transcendent_batches) if transcendent_batches else 0.0,
            "transcendent_updates": (
                int(transcendent_config.architect.total_updates)
                if transcendent_config is not None and transcendent_config.enabled
                else 0
            ),
            "rdrop_loss": (rdrop_loss_total / rdrop_batches) if rdrop_batches else 0.0,
            "rdrop_kl": (rdrop_kl_total / rdrop_batches) if rdrop_batches else 0.0,
            "rdrop_passes": float(rdrop_passes if rdrop_enabled else 0),
            "rdrop_alpha": float(rdrop_alpha if rdrop_enabled else 0.0),
            "frontier_loss": (frontier_loss_total / frontier_sample_total) if frontier_sample_total else 0.0,
            "frontier_novelty": (frontier_novelty_total / frontier_sample_total) if frontier_sample_total else 0.0,
            "frontier_abstraction": (frontier_abstraction_total / frontier_sample_total) if frontier_sample_total else 0.0,
            "frontier_transfer": (frontier_transfer_total / frontier_sample_total) if frontier_sample_total else 0.0,
            "frontier_curiosity": (frontier_curiosity_total / frontier_sample_total) if frontier_sample_total else 0.0,
            "frontier_emotion": (frontier_emotion_total / frontier_sample_total) if frontier_sample_total else 0.0,
            "frontier_meta": (frontier_meta_total / frontier_sample_total) if frontier_sample_total else 0.0,
            "frontier_diversity": (frontier_diversity_total / frontier_batches) if frontier_batches else 0.0,
            "frontier_updates": (
                int(frontier_config.catalyst.total_updates)
                if frontier_config is not None and frontier_config.enabled
                else 0
            ),
            "moe_loss": (moe_loss_total / moe_sample_total) if moe_sample_total else 0.0,
            "moe_entropy": (moe_entropy_total / moe_sample_total) if moe_sample_total else 0.0,
            "moe_entropy_gap": (moe_gap_total / moe_sample_total) if moe_sample_total else 0.0,
            "moe_balance": (moe_balance_total / moe_sample_total) if moe_sample_total else 0.0,
            "moe_active": (moe_active_total / moe_sample_total) if moe_sample_total else 0.0,
            "moe_max_gate": (moe_max_total / moe_sample_total) if moe_sample_total else 0.0,
            "moe_batches": float(moe_batches),
            "moe_utilisation_mean": (moe_util_mean_total / moe_batches) if moe_batches else 0.0,
            "moe_utilisation_min": (moe_util_min_total / moe_batches) if moe_batches else 0.0,
            "moe_utilisation_max": (moe_util_max_total / moe_batches) if moe_batches else 0.0,
            "duration": epoch_duration,
            "examples": float(total),
            "tokens": token_count_total if collect_performance_stats else 0.0,
            "batches": float(batch_counter),
            "micro_batches": float(micro_step_counter),
            "micro_batch_size": float(getattr(optimizer, "micro_batch_size", 0)),
        },
    )


def evaluate(
    model: nn.Module,
    dataloader: Iterable[Sequence[Any]],
    criterion: Callable[[TorchTensor, TorchTensor], TorchTensor],
    device: TorchDevice,
    *,
    return_details: bool = False,
    emotion_config: EmotionTrainingConfig | None = None,
    meta_stacker: MetaIntentStacker | None = None,
) -> tuple[float, float] | tuple[float, float, list[int], list[int], list[list[float]]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    device_type = getattr(device, "type", None)
    non_blocking = device_type in {"cuda", "mps"}
    detailed_targets: list[int] = []
    detailed_predictions: list[int] = []
    detailed_probabilities: list[list[float]] = []
    dataset_obj = getattr(dataloader, "dataset", None)
    dataset_has_emotion = bool(getattr(dataset_obj, "include_emotion", False))
    dataset_has_keywords = bool(getattr(dataset_obj, "include_keywords", False))
    with torch.no_grad():
        for batch_obj in dataloader:
            batch = tuple(batch_obj)
            if len(batch) < 5:
                raise ValueError("Evaluation batches must provide teacher logits even if empty.")
            inputs = _ensure_tensor(batch[0], device=device, non_blocking=non_blocking)
            targets = _ensure_tensor(batch[1], device=device, non_blocking=non_blocking)
            attention_mask = _ensure_tensor(batch[3], device=device, non_blocking=non_blocking)
            emotion_features: TorchTensor | None = None
            keyword_logits: TorchTensor | None = None
            next_index = 5
            if dataset_has_emotion and len(batch) > next_index:
                emotion_candidate = batch[next_index]
                if emotion_candidate is not None:
                    emotion_features = _ensure_tensor(
                        emotion_candidate,
                        device=device,
                        dtype=torch.float32,
                        non_blocking=non_blocking,
                    )
                    if emotion_features.dim() == 1:
                        emotion_features = emotion_features.unsqueeze(0)
                next_index += 1
            if dataset_has_keywords and len(batch) > next_index:
                keyword_candidate = batch[next_index]
                if keyword_candidate is not None:
                    keyword_logits = _ensure_tensor(
                        keyword_candidate,
                        device=device,
                        dtype=torch.float32,
                        non_blocking=non_blocking,
                    )
                    if keyword_logits.dim() == 1:
                        keyword_logits = keyword_logits.unsqueeze(0)
                    if keyword_logits.numel() == 0:
                        keyword_logits = None
            supports_emotion = (
                emotion_features is not None
                and emotion_features.numel() > 0
                and emotion_config is not None
                and emotion_config.enabled
                and bool(getattr(model, "supports_emotion_features", False))
            )

            logits: TorchTensor
            if supports_emotion:
                model_output = model(
                    inputs,
                    attention_mask=attention_mask,
                    emotion_features=emotion_features,
                    return_components=True,
                )
            else:
                model_output = model(inputs, attention_mask=attention_mask)
            if isinstance(model_output, tuple):
                logits = cast(TorchTensor, model_output[0])
            else:
                logits = cast(TorchTensor, model_output)
            base_logits = logits
            keyword_tensor: TorchTensor | None = None
            if (
                keyword_logits is not None
                and keyword_logits.shape[-1] == logits.shape[-1]
            ):
                keyword_tensor = keyword_logits
                logits = logits + keyword_tensor
            if meta_stacker is not None:
                meta_rows: list[list[float]] = []
                has_adjustment = False
                batch_size = int(base_logits.size(0))
                for row_idx in range(batch_size):
                    keyword_row = None
                    if keyword_tensor is not None and row_idx < int(keyword_tensor.size(0)):
                        keyword_row = keyword_tensor[row_idx]
                    adjustment = meta_stacker.compute_adjustment(
                        base_logits[row_idx],
                        keyword_row,
                    )
                    if adjustment:
                        has_adjustment = True
                        meta_rows.append(adjustment)
                    else:
                        meta_rows.append([0.0] * int(base_logits.size(-1)))
                if has_adjustment:
                    meta_tensor = torch.tensor(
                        meta_rows,
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                    logits = logits + meta_tensor
            loss_values = criterion(logits, targets)
            if loss_values.dim() == 0:
                loss_values = loss_values.unsqueeze(0)

            batch_loss = float(loss_values.mean().item())
            total_loss += batch_loss * int(targets.size(0))
            predictions = logits.argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.size(0))
            detailed_targets.extend(targets.cpu().tolist())
            detailed_predictions.extend(predictions.cpu().tolist())
            detailed_probabilities.extend(
                torch.softmax(logits, dim=-1).cpu().tolist()
            )
    if return_details:
        return (
            total_loss / max(total, 1),
            correct / max(total, 1),
            detailed_targets,
            detailed_predictions,
            detailed_probabilities,
        )
    return total_loss / max(total, 1), correct / max(total, 1)

def _optimizer_supports_momentum(optimizer: TorchOptimizer) -> bool:
    """Return ``True`` when the optimizer exposes momentum/beta settings."""

    def _has_momentum(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, (tuple, list)):
            for item in cast(Iterable[object], value):
                if _has_momentum(item):
                    return True
            return False
        return True

    def _mapping_has_momentum(settings: Mapping[str, object]) -> bool:
        if "momentum" in settings and _has_momentum(settings.get("momentum")):
            return True
        betas = settings.get("betas")
        if _has_momentum(betas):
            return True
        if "beta1" in settings and _has_momentum(settings.get("beta1")):
            return True
        return False

    defaults_obj = getattr(optimizer, "defaults", None)
    if isinstance(defaults_obj, Mapping):
        defaults = cast(Mapping[str, object], defaults_obj)
        if _mapping_has_momentum(defaults):
            return True
    param_groups_obj = getattr(optimizer, "param_groups", None)
    if isinstance(param_groups_obj, Iterable):
        for group_obj in cast(Iterable[object], param_groups_obj):
            if not isinstance(group_obj, Mapping):
                continue
            group = cast(Mapping[str, object], group_obj)
            if _mapping_has_momentum(group):
                return True
    return False


def create_scheduler(
    optimizer: TorchOptimizer,
    scheduler_type: str,
    epochs: int,
    steps_per_epoch: int,
    max_lr: float,
) -> tuple[SchedulerLike | None, bool]:
    if scheduler_type == "onecycle" and epochs > 0 and steps_per_epoch > 0:
        cycle_momentum = _optimizer_supports_momentum(optimizer)
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
            cycle_momentum=cycle_momentum,
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
    vocab: dict[str, int],
    label_to_idx: dict[str, int],
    max_len: int,
    device: TorchDevice,
    threshold: float,
    vocab_config: VocabularyConfig | None = None,
    tokenizer: Any | None = None,
    tokenizer_cache: Callable[[str], tuple[Sequence[int], Sequence[int]]] | None = None,
    embedding_model: Callable[[str], VectorLike] | None = None,
    emotion_encoder: EmotionLexicon | None = None,
    emotion_dim: int = 0,
    emotion_config: EmotionTrainingConfig | None = None,
    consistency_passes: int = 1,
    consistency_max_std: float = 0.08,
    consistency_min_agreement: float = 0.6,
    metadata_encoder: StructuredMetadataEncoder | None = None,
    lexicon_dim: int = 0,
    metadata_dim: int = 0,
    keyword_calibrator: KeywordIntentCalibrator | None = None,
    symbolic_router: CognitiveIntentRouter | None = None,
    meta_stacker: MetaIntentStacker | None = None,
) -> tuple[list[PseudoLabelDecision], list[str], dict[str, Any]]:
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    decisions: list[PseudoLabelDecision] = []
    remaining: list[str] = []

    passes = max(1, int(consistency_passes))
    raw_max_std = float(consistency_max_std)
    effective_max_std = float("inf") if passes <= 1 else (raw_max_std if raw_max_std > 0 else float("inf"))
    min_agreement = max(0.0, min(consistency_min_agreement, 1.0))

    total_candidates = 0
    accepted_candidates = 0
    confidence_sum = 0.0
    std_sum = 0.0
    agreement_sum = 0.0
    consistency_sum = 0.0
    std_sum_all = 0.0
    agreement_sum_all = 0.0
    reject_confidence = 0
    reject_consistency = 0
    reject_agreement = 0

    previous_mode = model.training
    if passes > 1:
        model.train()
    else:
        model.eval()
    try:
        with torch.no_grad():
            for text in texts:
                total_candidates += 1
                ids, mask, emotion_features = _prepare_model_inputs(
                    text,
                    vocab=vocab,
                    max_len=max_len,
                    device=device,
                    tokenizer=tokenizer,
                    tokenizer_cache=tokenizer_cache,
                    embedding_model=embedding_model,
                    vocab_config=vocab_config,
                    emotion_encoder=emotion_encoder,
                    emotion_dim=emotion_dim,
                    metadata=None,
                    metadata_encoder=metadata_encoder,
                    lexicon_dim=lexicon_dim,
                    metadata_dim=metadata_dim,
                )
                adjustment_vector = compose_logit_adjustments(
                    text,
                    calibrator=keyword_calibrator,
                    router=symbolic_router,
                )
                supports_emotion = (
                    emotion_features is not None
                    and emotion_features.numel() > 0
                    and getattr(model, "supports_emotion_features", False)
                    and (emotion_config is None or emotion_config.enabled)
                )
                probability_passes: list[TorchTensor] = []
                effective_passes = max(1, passes)
                logits: TorchTensor = torch.empty(0, dtype=ids.dtype, device=ids.device)
                for _ in range(effective_passes):
                    if supports_emotion:
                        output = model(
                            ids,
                            attention_mask=mask,
                            emotion_features=emotion_features,
                            return_components=True,
                        )
                    else:
                        output = model(ids, attention_mask=mask)
                    if isinstance(output, tuple):
                        logits = cast(TorchTensor, output[0])
                    else:
                        logits = cast(TorchTensor, output)
                base_logits = logits
                keyword_tensor = None
                if adjustment_vector:
                    keyword_tensor = torch.tensor(
                        adjustment_vector,
                        dtype=logits.dtype,
                        device=logits.device,
                    ).unsqueeze(0)
                    logits = logits + keyword_tensor
                if meta_stacker is not None:
                    meta_adjustment = meta_stacker.compute_adjustment(
                        base_logits,
                        keyword_tensor[0] if keyword_tensor is not None else None,
                    )
                    if meta_adjustment:
                        meta_tensor = torch.tensor(
                            meta_adjustment,
                            dtype=logits.dtype,
                            device=logits.device,
                        ).unsqueeze(0)
                        logits = logits + meta_tensor
                    probs = torch.softmax(logits, dim=-1)
                    probability_passes.append(probs.detach())
                stacked = torch.stack(probability_passes, dim=0)
                averaged = stacked.mean(dim=0)
                confidence_tensor, predicted = averaged.max(dim=-1)
                predicted_idx = predicted.item()
                if passes > 1:
                    predicted_distribution = stacked[:, 0, predicted_idx]
                    std_value = float(predicted_distribution.std(unbiased=False).item())
                    pass_predictions = stacked.argmax(dim=-1).view(passes)
                    agreement = float((pass_predictions == predicted_idx).float().mean().item())
                else:
                    std_value = 0.0
                    agreement = 1.0
                std_sum_all += std_value
                agreement_sum_all += agreement
                score = float(confidence_tensor.item())
                std_ok = std_value <= effective_max_std
                agreement_ok = agreement >= min_agreement if passes > 1 else True
                if score >= threshold and std_ok and agreement_ok:
                    label = idx_to_label[predicted_idx]
                    consistency_score = compute_consistency_score(agreement, std_value, effective_max_std)
                    decisions.append(
                        PseudoLabelDecision(
                            text=text,
                            label=label,
                            confidence=score,
                            consistency=consistency_score,
                            agreement=agreement,
                            std=std_value,
                        )
                    )
                    accepted_candidates += 1
                    confidence_sum += score
                    std_sum += std_value
                    agreement_sum += agreement
                    consistency_sum += consistency_score
                else:
                    remaining.append(text)
                    if score < threshold:
                        reject_confidence += 1
                    if passes > 1 and not std_ok:
                        reject_consistency += 1
                    if passes > 1 and not agreement_ok:
                        reject_agreement += 1
    finally:
        model.train(previous_mode)

    stats: dict[str, Any] = {
        "evaluated": float(total_candidates),
        "accepted": float(accepted_candidates),
        "avg_confidence": confidence_sum / accepted_candidates if accepted_candidates else 0.0,
        "avg_std": std_sum / accepted_candidates if accepted_candidates else 0.0,
        "avg_agreement": agreement_sum / accepted_candidates if accepted_candidates else 0.0,
        "avg_consistency": consistency_sum / accepted_candidates if accepted_candidates else 0.0,
        "avg_std_all": std_sum_all / total_candidates if total_candidates else 0.0,
        "avg_agreement_all": agreement_sum_all / total_candidates if total_candidates else 0.0,
        "reject_confidence": float(reject_confidence),
        "reject_consistency": float(reject_consistency),
        "reject_agreement": float(reject_agreement),
        "passes": float(passes),
        "max_std": float(raw_max_std if passes > 1 else 0.0),
        "min_agreement": float(min_agreement if passes > 1 else 1.0),
    }
    return decisions, remaining, stats


@dataclass
class ModelPrediction:
    label: str
    confidence: float
    top_predictions: list[tuple[str, float]]


def _prepare_model_inputs(
    text: str,
    *,
    vocab: dict[str, int],
    max_len: int,
    device: TorchDevice,
    tokenizer: Any | None = None,
    tokenizer_cache: Callable[[str], tuple[Sequence[int], Sequence[int]]] | None = None,
    embedding_model: Callable[[str], VectorLike] | None = None,
    vocab_config: VocabularyConfig | None = None,
    emotion_encoder: EmotionLexicon | None = None,
    emotion_dim: int = 0,
    metadata: Mapping[str, str] | None = None,
    metadata_encoder: StructuredMetadataEncoder | None = None,
    lexicon_dim: int | None = None,
    metadata_dim: int | None = None,
) -> tuple[TorchTensor, TorchTensor, TorchTensor | None]:
    emotion_tensor: TorchTensor | None = None
    if embedding_model is not None:
        vector = embedding_model(text)
        if torch.is_tensor(vector):
            tensor = cast(TorchTensor, vector).detach()
            ids = tensor.to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(0)
        else:
            ids = torch.tensor(vector, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.ones_like(ids, dtype=torch.float32)
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
        raw_ids = encode_text(text, vocab, max_len, config=vocab_config)
        ids = torch.tensor(raw_ids, dtype=torch.long, device=device).unsqueeze(0)
        pad_idx = vocab.get(PAD_TOKEN, 0)
        mask_values = [1 if token != pad_idx else 0 for token in raw_ids]
        mask = torch.tensor(mask_values, dtype=torch.long, device=device).unsqueeze(0)
    active_lexicon = emotion_encoder if lexicon_dim is None or (lexicon_dim and lexicon_dim > 0) else None
    resolved_lexicon_dim = 0
    if active_lexicon is not None:
        resolved_lexicon_dim = lexicon_dim if lexicon_dim is not None else len(active_lexicon.emotions)
    resolved_metadata_dim = metadata_dim if metadata_dim is not None else max(0, emotion_dim - resolved_lexicon_dim)
    total_dim = emotion_dim if emotion_dim > 0 else resolved_lexicon_dim + resolved_metadata_dim
    if total_dim > 0 and (active_lexicon is not None or metadata_encoder is not None):
        values = compose_emotion_features(
            text,
            metadata,
            lexicon=active_lexicon,
            metadata_encoder=metadata_encoder,
            lexicon_dim=resolved_lexicon_dim,
            metadata_dim=resolved_metadata_dim,
        )
        if len(values) < total_dim:
            values.extend([0.0] * (total_dim - len(values)))
        elif len(values) > total_dim:
            values = values[:total_dim]
        emotion_tensor = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
    return ids, mask, emotion_tensor


def predict_with_trace(
    model: nn.Module,
    text: str,
    *,
    vocab: dict[str, int],
    label_to_idx: dict[str, int],
    max_len: int,
    device: TorchDevice,
    tokenizer: Any | None = None,
    tokenizer_cache: Callable[[str], tuple[Sequence[int], Sequence[int]]] | None = None,
    embedding_model: Callable[[str], VectorLike] | None = None,
    top_k: int = 3,
    vocab_config: VocabularyConfig | None = None,
    emotion_encoder: EmotionLexicon | None = None,
    emotion_dim: int = 0,
    emotion_config: EmotionTrainingConfig | None = None,
    metadata: Mapping[str, str] | None = None,
    metadata_encoder: StructuredMetadataEncoder | None = None,
    lexicon_dim: int = 0,
    metadata_dim: int = 0,
    calibrator: KeywordIntentCalibrator | None = None,
    symbolic_router: CognitiveIntentRouter | None = None,
    meta_stacker: MetaIntentStacker | None = None,
) -> ModelPrediction:
    ids, mask, emotion_features = _prepare_model_inputs(
        text,
        vocab=vocab,
        max_len=max_len,
        device=device,
        tokenizer=tokenizer,
        tokenizer_cache=tokenizer_cache,
        embedding_model=embedding_model,
        vocab_config=vocab_config,
        emotion_encoder=emotion_encoder,
        emotion_dim=emotion_dim,
        metadata=metadata,
        metadata_encoder=metadata_encoder,
        lexicon_dim=lexicon_dim,
        metadata_dim=metadata_dim,
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
            output = model(
                ids,
                attention_mask=mask,
                emotion_features=emotion_features,
                return_components=True,
            )
        else:
            output = model(ids, attention_mask=mask)
        if isinstance(output, tuple):
            logits = cast(TorchTensor, output[0])
        else:
            logits = cast(TorchTensor, output)
        base_logits = logits
        adjustment_vector = compose_logit_adjustments(
            text,
            calibrator=calibrator,
            router=symbolic_router,
        )
        keyword_tensor: TorchTensor | None = None
        if adjustment_vector:
            keyword_tensor = torch.tensor(
                adjustment_vector,
                dtype=logits.dtype,
                device=logits.device,
            ).unsqueeze(0)
            logits = logits + keyword_tensor
        if meta_stacker is not None:
            base_row = base_logits[0] if base_logits.dim() > 1 else base_logits
            keyword_row = None
            if keyword_tensor is not None:
                keyword_row = keyword_tensor[0] if keyword_tensor.dim() > 1 else keyword_tensor
            meta_adjustment = meta_stacker.compute_adjustment(
                base_row,
                keyword_row,
            )
            if meta_adjustment:
                meta_tensor = torch.tensor(
                    meta_adjustment,
                    dtype=logits.dtype,
                    device=logits.device,
                ).unsqueeze(0)
                logits = logits + meta_tensor
        probs = torch.softmax(logits, dim=-1)
        confidence_tensor, predicted = probs.max(dim=1)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    label = idx_to_label[predicted.item()]
    confidence = float(confidence_tensor.item())
    ranked: list[tuple[str, float]] = []
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
    vocab: dict[str, int],
    label_to_idx: dict[str, int],
    max_len: int,
    device: TorchDevice,
    tokenizer: Any | None = None,
    tokenizer_cache: Callable[[str], tuple[Sequence[int], Sequence[int]]] | None = None,
    embedding_model: Callable[[str], VectorLike] | None = None,
    return_confidence: bool = False,
    vocab_config: VocabularyConfig | None = None,
    emotion_encoder: EmotionLexicon | None = None,
    emotion_dim: int = 0,
    emotion_config: EmotionTrainingConfig | None = None,
    metadata: Mapping[str, str] | None = None,
    metadata_encoder: StructuredMetadataEncoder | None = None,
    lexicon_dim: int = 0,
    metadata_dim: int = 0,
    calibrator: KeywordIntentCalibrator | None = None,
    symbolic_router: CognitiveIntentRouter | None = None,
    meta_stacker: MetaIntentStacker | None = None,
) -> str | tuple[str, float]:
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
        vocab_config=vocab_config,
        emotion_encoder=emotion_encoder,
        emotion_dim=emotion_dim,
        emotion_config=emotion_config,
        metadata=metadata,
        metadata_encoder=metadata_encoder,
        lexicon_dim=lexicon_dim,
        metadata_dim=metadata_dim,
        calibrator=calibrator,
        symbolic_router=symbolic_router,
        meta_stacker=meta_stacker,
    )
    if return_confidence:
        return prediction.label, prediction.confidence
    return prediction.label


@dataclass
class ResponseOutcome:
    message: str
    strategy: str
    basis: str | None = None


QUESTION_STOPWORDS: set[str] = {
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
    focus_words: list[str] = []
    for word in words:
        if normalise_text(word) in QUESTION_STOPWORDS:
            continue
        focus_words.append(word)
    focus = " ".join(focus_words).strip()
    return _truncate_snippet(focus)


def _humanize_label(label: str) -> str:
    return label.replace("_", " ")
def answer_question(text: str) -> ResponseOutcome:
    stripped = text.strip()
    cleaned = _collapse_whitespace(stripped)
    if not cleaned:
        return ResponseOutcome(
            message="I didn't catch the question—could you rephrase it so I can help?",
            strategy="question_clarification",
        )
    lowered = normalise_text(text)
    rng = _orion_seed_rng(f"response::question::{cleaned}")
    features = inspect_text_characteristics(cleaned)
    focus = _question_focus(stripped)
    target_phrase = focus or "this question"
    segments: list[str] = []
    segments.append(_describe_punctuation(features, rng))
    segments.append(_describe_case_usage(features, rng))
    segments.append(_describe_question_signature(features, rng))
    seduction_line = _describe_seduction(features, rng)
    if seduction_line:
        segments.append(seduction_line)
    segments.append(f"Focus converges on {target_phrase}; Orion keeps the thread taut.")
    strategies: list[tuple[str, str]] = []
    strategies.append(("general_research", f"I'll explore {target_phrase} and surface a grounded answer."))
    if "remind" in lowered or "reminder" in lowered:
        strategies.append(("reminder_preparation", f"I'll anchor a reminder around {target_phrase} and time the follow-up."))
    if re.search(r"\b(when|what time)\b", lowered):
        strategies.append(("schedule_lookup", f"I'll confirm the timing woven into {target_phrase} and broadcast it back."))
    if re.search(r"\bwhere\b", lowered):
        strategies.append(("location_lookup", f"I'll locate the destination implied by {target_phrase} and share the path."))
    if re.search(r"\bwho\b", lowered):
        strategies.append(("ownership_lookup", f"I'll trace the ownership of {target_phrase} and connect you."))
    if lowered.startswith("how can i") or lowered.startswith("how to ") or "how do i" in lowered:
        strategies.append(("process_guidance", f"I'll sketch the steps that let {target_phrase} unfold."))
    if "why" in lowered:
        strategies.append(("root_cause_investigation", f"I'll investigate why {target_phrase} is happening and distil the cause."))
    if any(keyword in lowered for keyword in ["backup", "server", "database", "credential", "outage", "incident"]):
        strategies.append(("operations_follow_up", f"I'll cross-check operational notes about {target_phrase} and coordinate recovery."))
    if any(keyword in lowered for keyword in ["upload", "share", "send", "submit", "post"]):
        strategies.append(("handoff_lookup", f"I'll verify where {target_phrase} should land and relay the hand-off route."))
    if any(keyword in lowered for keyword in ["issue", "problem", "broken", "crash", "error", "not working"]):
        strategies.append(("incident_triage", f"I'll triage the incident around {target_phrase} and keep the status visible."))
    chosen_strategy, plan_line = strategies[-1]
    segments.append(plan_line)
    reflections = craft_orion_reflections(
        features,
        label="question",
        rng=_orion_seed_rng(f"response::question::reflections::{cleaned}"),
        context="response",
    )
    if reflections:
        segments.extend(reflections[:2])
    message_segments = [segment for segment in segments if segment]
    message = " ".join(message_segments)
    basis = focus or _truncate_snippet(cleaned) or None
    return ResponseOutcome(message=message, strategy=chosen_strategy, basis=basis)


def _describe_punctuation(features: Mapping[str, Any], rng: random.Random) -> str:
    comma_count = int(features.get("comma_count", 0))
    question_marks = int(features.get("question_marks", 0))
    exclamation_marks = int(features.get("exclamation_marks", 0))
    pauses_word = rng.choice(["pauses", "breaths", "rests"])
    sparks_word = rng.choice(["sparks", "flashes", "signals"])
    fragments: list[str] = []
    if comma_count == 0:
        fragments.append("No commas appear, so the flow rushes without marked pauses.")
    elif comma_count == 1:
        fragments.append("A single comma interrupts the stream, carving one deliberate pause.")
    else:
        fragments.append(f"{comma_count} commas sketch out {pauses_word} that pace the rhythm.")
    if question_marks > 0:
        suffix = "?" if question_marks == 1 else "?s"
        fragments.append(f"{question_marks} question mark{'' if question_marks == 1 else 's'} {sparks_word} inquiry{suffix}.")
    if exclamation_marks > 0:
        intensity = rng.choice(["surges", "bursts", "surges"])
        fragments.append(f"{exclamation_marks} exclamation burst{'' if exclamation_marks == 1 else 's'} {intensity} across the tone.")
    return " ".join(fragments)


def _describe_case_usage(features: Mapping[str, Any], rng: random.Random) -> str:
    uppercase_ratio = float(features.get("uppercase_ratio", 0.0))
    uppercase_tokens_obj = features.get("uppercase_tokens")
    uppercase_tokens: list[str] = []
    if isinstance(uppercase_tokens_obj, Sequence):
        for item in cast(Sequence[object], uppercase_tokens_obj):
            token = str(item)
            if token:
                uppercase_tokens.append(token)
    if uppercase_ratio < 0.05:
        mood = rng.choice(["soft", "even", "gentle"])
        return f"Letter casing stays {mood}, so Orion leans in to catch nuance."
    if uppercase_ratio < 0.2:
        if uppercase_tokens:
            sample = ", ".join(uppercase_tokens[:3])
            return f"Occasional uppercase words ({sample}) glint through, hinting at emphasis."
        return "Uppercase letters flicker quietly, hinting at subtle emphasis."
    if uppercase_tokens:
        sample = ", ".join(uppercase_tokens[:3])
        return f"Uppercase intensity spikes—{sample} flare like constellations calling for attention."
    return "Uppercase saturation rises, so Orion balances the energy with calm intent."


def _describe_question_signature(features: Mapping[str, Any], rng: random.Random) -> str:
    question_type = str(features.get("question_type", "statement"))
    false_question = bool(features.get("false_question"))
    trailing_cue = bool(features.get("trailing_question_cue"))
    if question_type == "statement":
        return rng.choice(
            [
                "It carries statement energy; Orion reads it as a declaration to absorb.",
                "No question pulse detected, so Orion listens for implied needs.",
            ]
        )
    if question_type == "direct":
        return rng.choice(
            [
                "The wording lands as a direct question, inviting an anchored reply.",
                "Direct interrogative form detected; Orion prepares a clear answer.",
            ]
        )
    if question_type == "emotional":
        return "Punctuation softens into emotional questioning—Orion steadies before responding."
    if false_question or trailing_cue:
        return "It dresses like a question yet leans rhetorical, so Orion checks whether help is truly desired."
    return "Question signals appear unusual; Orion cross-checks intent before answering."


def _describe_seduction(features: Mapping[str, Any], rng: random.Random) -> str | None:
    seduction_style = str(features.get("seduction_style", "none"))
    seduction_terms = list(features.get("seduction_terms") or [])
    if seduction_style == "none":
        return None
    palette = {
        "warm": rng.choice(["gentle warmth", "soft charm", "kind affection"]),
        "coaxing": rng.choice(["coaxing shimmer", "swaying allure", "careful persuasion"]),
        "intense": rng.choice(["bold magnetism", "fierce allure", "bright temptation"]),
    }
    mood = palette.get(seduction_style, "unusual charm")
    if seduction_terms:
        sample = ", ".join(sorted(set(seduction_terms))[:3])
        return f"Terms like {sample} add {mood}; Orion separates signal from sentiment."
    return f"A {mood} rides the phrasing; Orion stays curious and grounded."


def _describe_intent_landscape(label: str, rng: random.Random) -> str:
    human_label = _humanize_label(label)
    textures = [
        "a collaborative orbit",
        "a pragmatic corridor",
        "an empathetic field",
        "a creative vector",
        "a decisive channel",
        "an investigative lane",
    ]
    texture = rng.choice(textures)
    return f"Intent traces toward {human_label}; Orion frames it inside {texture}."


def _describe_response_plan(label: str, features: Mapping[str, Any], rng: random.Random) -> str:
    verbs = ["map", "trace", "synthesize", "prototype", "document", "stabilize", "illuminate", "follow through on"]
    verb = rng.choice(verbs)
    human_label = _humanize_label(label)
    phrases = [
        f"{verb} the {human_label} path and report the movement back.",
        f"{verb} how {human_label} energy should unfold next.",
        f"{verb} the steps that let {human_label} intentions become real.",
    ]
    if bool(features.get("false_question")):
        phrases.append(
            f"{verb} whether the {human_label} thread hides a question or simply seeks resonance."
        )
    if bool(features.get("false_seduction")):
        phrases.append(
            f"{verb} the spine of the {human_label} request while filtering out charming detours."
        )
    if bool(features.get("likely_question")):
        phrases.append(
            f"{verb} the answer that lets this {human_label} question land with clarity."
        )
    return rng.choice(phrases)


def generate_response(label: str, text: str) -> ResponseOutcome:
    if label == "question":
        return answer_question(text)
    cleaned = _collapse_whitespace(text.strip())
    if not cleaned:
        return ResponseOutcome(
            message="The input felt empty, so Orion waits for clearer language before acting.",
            strategy=f"orion_dynamic::{label}",
        )
    rng = _orion_seed_rng(f"response::{label}::{cleaned}")
    features = inspect_text_characteristics(cleaned)
    reflections = craft_orion_reflections(
        features,
        label=label,
        rng=_orion_seed_rng(f"response::reflections::{label}::{cleaned}"),
        context="response",
    )
    segments: list[str] = []
    segments.append(_describe_punctuation(features, rng))
    segments.append(_describe_case_usage(features, rng))
    segments.append(_describe_question_signature(features, rng))
    seduction_line = _describe_seduction(features, rng)
    if seduction_line:
        segments.append(seduction_line)
    segments.append(_describe_intent_landscape(label, rng))
    segments.append(_describe_response_plan(label, features, rng))
    if reflections:
        segments.extend(reflections[:2])
    message = " ".join(segment for segment in segments if segment)
    basis = _truncate_snippet(cleaned) or None
    strategy = f"orion_dynamic::{label}"
    return ResponseOutcome(message=message, strategy=strategy, basis=basis)


def write_accuracy_readme(
    target_dir: Path,
    model_name: str,
    metrics: dict[str, Any],
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
    metadata: dict[str, Any],
    metrics: dict[str, Any],
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
    parser.add_argument("--encoder-type", choices=["bilstm", "transformer", "st"], default="bilstm",
                        help="Select between the BiLSTM encoder and a pretrained transformer backbone.")
    parser.add_argument("--transformer-model", type=str, default="distilbert-base-uncased",
                        help="Hugging Face model checkpoint to fine-tune when --encoder-type=transformer.")
    parser.add_argument("--transformer-learning-rate", type=float, default=1e-4,
                        help="Learning rate used when fine-tuning the transformer encoder.")
    parser.add_argument("--transformer-layerwise-decay", type=float, default=0.95,
                        help="Layer-wise learning-rate decay applied to transformer encoders (set to 1.0 to disable).")
    parser.add_argument("--sentence-transformer-model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformer checkpoint for frozen-embedding classification when --encoder-type=st.")
    parser.add_argument("--st-hidden-dim", type=int, default=512,
                        help="Hidden dimension of the feed-forward head for sentence-transformer embeddings.")
    parser.add_argument("--st-dropout", type=float, default=0.2,
                        help="Dropout applied in the sentence-transformer feed-forward head.")
    parser.add_argument("--st-mlp-layers", type=int, default=1,
                        help="Number of hidden layers in the sentence-transformer classification head (>=1).")
    parser.add_argument("--st-mlp-expansion", type=float, default=1.0,
                        help="Expansion factor applied after each hidden layer when --st-mlp-layers>1 (>=1.0).")
    parser.add_argument("--st-mlp-activation", choices=["relu", "gelu", "silu", "tanh"], default="relu",
                        help="Activation function used inside the sentence-transformer classification head.")
    parser.add_argument("--st-mlp-layer-norm", action="store_true",
                        help="Apply LayerNorm after each hidden projection in the sentence-transformer head.")
    parser.add_argument("--st-mlp-residual", action="store_true",
                        help="Enable residual skip connections when consecutive layers share the same width.")
    parser.add_argument("--st-final-dropout", type=float, default=0.0,
                        help="Dropout applied after the final hidden layer in the sentence-transformer head.")
    parser.add_argument("--st-moe-experts", type=int, default=0,
                        help="Number of soft experts in the sentence-transformer head (>=2 enables the mixture).")
    parser.add_argument("--st-moe-hidden-dim", type=int, default=0,
                        help="Hidden width for each expert projection in the mixture (0 defaults to the head width).")
    parser.add_argument("--st-moe-activation", choices=["relu", "gelu", "silu", "tanh"], default="gelu",
                        help="Activation applied inside each mixture expert (defaults to GELU).")
    parser.add_argument("--st-moe-dropout", type=float, default=0.1,
                        help="Dropout applied within each mixture expert block.")
    parser.add_argument("--st-moe-temperature", type=float, default=1.0,
                        help="Gating temperature for the mixture-of-experts head (>0).")
    parser.add_argument("--st-moe-topk", type=int, default=0,
                        help="Optional top-k routing for the mixture-of-experts gate (0 uses all experts).")
    parser.add_argument("--st-moe-entropy-weight", type=float, default=0.05,
                        help="Regularisation weight encouraging high-entropy expert utilisation.")
    parser.add_argument("--st-moe-balance-weight", type=float, default=0.05,
                        help="Regularisation weight encouraging uniform expert utilisation.")
    parser.add_argument("--st-moe-layer-norm", action="store_true",
                        help="Apply layer normalisation inside each mixture expert block.")
    parser.add_argument("--st-moe-utilisation-momentum", type=float, default=0.9,
                        help="Momentum used when tracking running expert utilisation statistics (in [0,1)).")
    parser.add_argument("--embedding-dim", type=int, default=320,
                        help="Size of the token embeddings (BiLSTM encoder only).")
    parser.add_argument("--hidden-dim", type=int, default=384,
                        help="Hidden dimension of the BiLSTM encoder.")
    parser.add_argument("--ffn-dim", type=int, default=768,
                        help="Width of the feed-forward layer after attention (BiLSTM encoder only).")
    parser.add_argument("--encoder-layers", type=int, default=3,
                        help="Number of stacked BiLSTM layers.")
    parser.add_argument("--attention-heads", type=int, default=8,
                        help="Number of attention heads for the self-attention block (BiLSTM encoder only).")
    parser.add_argument("--dropout", type=float, default=0.25,
                        help="Dropout rate applied throughout the network (BiLSTM encoder only).")
    parser.add_argument("--bilstm-conv-head", dest="bilstm_conv_head", action="store_true",
                        help="Enable the multi-scale convolutional head that augments the BiLSTM encoder.")
    parser.add_argument("--no-bilstm-conv-head", dest="bilstm_conv_head", action="store_false",
                        help="Disable the multi-scale convolutional head for the BiLSTM encoder.")
    parser.set_defaults(bilstm_conv_head=True)
    parser.add_argument("--bilstm-conv-kernels", type=str, default="3,5,7",
                        help="Comma-separated kernel sizes used by the convolutional head (BiLSTM encoder only).")
    parser.add_argument("--bilstm-conv-channels", type=int, default=256,
                        help="Number of channels produced by each convolution in the head (BiLSTM encoder only).")
    parser.add_argument("--bilstm-conv-dropout", type=float, default=0.2,
                        help="Dropout applied inside the convolutional head (BiLSTM encoder only).")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of supervised training epochs.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size.")
    parser.add_argument("--micro-batch-size", type=int, default=0,
                        help="Optional micro-batch size used to split each loader batch; 0 enables auto-selection.")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Peak learning rate for the optimiser/scheduler (BiLSTM encoder).")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay used by AdamW.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Clip gradients to this norm (set <=0 to disable).")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Amount of label smoothing to apply during training.")
    parser.add_argument("--rdrop-alpha", type=float, default=0.0,
                        help="Weight applied to the symmetric KL penalty used for R-Drop regularisation (0 disables).")
    parser.add_argument("--rdrop-forward-passes", type=int, default=2,
                        help="Number of stochastic forward passes used to compute the R-Drop loss (>=2).")
    parser.add_argument("--scheduler", choices=["onecycle", "cosine", "none"], default="onecycle",
                        help="Learning-rate scheduler strategy.")
    parser.add_argument("--grad-accumulation-steps", type=int, default=4,
                        help="Number of gradient accumulation steps per optimiser update.")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="Exponential moving-average decay applied to model weights (0 disables).")
    parser.add_argument("--ema-start-epoch", type=int, default=1,
                        help="Global epoch after which EMA updates begin to accumulate.")
    parser.add_argument("--ema-use-for-eval", action="store_true",
                        help="Use EMA weights for validation, pseudo-labelling, and promotion once the EMA is active.")
    parser.add_argument("--ema-offload-cpu", dest="ema_offload_cpu", action="store_true",
                        help="Keep EMA weights on CPU, loading them to the accelerator only for evaluation.")
    parser.add_argument("--ema-keep-on-device", dest="ema_offload_cpu", action="store_false",
                        help="Retain EMA weights on the active training device instead of offloading to CPU.")
    parser.set_defaults(ema_offload_cpu=True)
    parser.add_argument("--ema-offload-dtype", type=str, default="float32",
                        help="Floating-point dtype used when storing offloaded EMA weights (e.g., float32, float16, bfloat16).")
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
    parser.add_argument("--min-freq", type=int, default=5,
                        help="Minimum token frequency required to enter the vocabulary.")
    parser.add_argument("--vocab-max-size", type=int, default=75000,
                        help="Maximum number of tokens retained in the vocabulary (0 disables the cap).")
    parser.add_argument("--vocab-extra-corpus", type=str, default="unlabeled,metadata",
                        help="Comma-separated list of extra text sources to enrich the vocabulary (options: unlabeled, metadata, none).")
    parser.add_argument("--vocab-include-bigrams", dest="vocab_include_bigrams", action="store_true",
                        help="Augment the vocabulary with word bigrams.")
    parser.add_argument("--no-vocab-bigrams", dest="vocab_include_bigrams", action="store_false",
                        help="Disable word bigram augmentation when building the vocabulary.")
    parser.set_defaults(vocab_include_bigrams=True)
    parser.add_argument("--vocab-include-trigrams", dest="vocab_include_trigrams", action="store_true",
                        help="Augment the vocabulary with word trigrams.")
    parser.add_argument("--no-vocab-trigrams", dest="vocab_include_trigrams", action="store_false",
                        help="Disable word trigram augmentation when building the vocabulary.")
    parser.set_defaults(vocab_include_trigrams=False)
    parser.add_argument("--vocab-include-char-ngrams", dest="vocab_include_char_ngrams", action="store_true",
                        help="Augment the vocabulary with character n-gram features.")
    parser.add_argument("--no-vocab-char-ngrams", dest="vocab_include_char_ngrams", action="store_false",
                        help="Disable character n-gram augmentation when building the vocabulary.")
    parser.set_defaults(vocab_include_char_ngrams=True)
    parser.add_argument("--vocab-char-ngram-min", type=int, default=3,
                        help="Minimum character n-gram size when char n-grams are enabled (>=1).")
    parser.add_argument("--vocab-char-ngram-max", type=int, default=5,
                        help="Maximum character n-gram size when char n-grams are enabled (>= min).")
    parser.add_argument("--vocab-char-ngram-limit", type=int, default=3,
                        help="Maximum number of character n-grams retained per token (0 keeps all).")
    parser.add_argument("--max-seq-len", type=int, default=128,
                        help="Maximum number of tokens retained per example (after tokenisation).")
    parser.add_argument("--auto-extend-max-seq", dest="auto_extend_max_seq", action="store_true",
                        help="Allow the trainer to increase max sequence length automatically when augmented tokens exceed the limit.")
    parser.add_argument("--no-auto-extend-max-seq", dest="auto_extend_max_seq", action="store_false",
                        help="Disable automatic sequence-length extension even when augmented tokens exceed the configured limit.")
    parser.set_defaults(auto_extend_max_seq=True)
    parser.add_argument("--self-train-rounds", type=int, default=10,
                        help="Number of self-training refinement rounds.")
    parser.add_argument("--self-train-epochs", type=int, default=2,
                        help="Additional epochs to run after adding pseudo-labelled data in each round.")
    parser.add_argument("--self-train-threshold", type=float, default=0.92,
                        help="Confidence threshold for accepting pseudo-labels.")
    parser.add_argument("--self-train-weight", type=float, default=0.5,
                        help="Loss weight applied to pseudo-labelled examples (relative to 1.0 for gold labels).")
    parser.add_argument("--self-train-consistency-passes", type=int, default=4,
                        help="Number of Monte Carlo passes used when gauging pseudo-label consistency (>=1).")
    parser.add_argument("--self-train-consistency-max-std", type=float, default=0.075,
                        help="Maximum allowed standard deviation of the predicted class probability across consistency passes.")
    parser.add_argument("--self-train-consistency-min-agreement", type=float, default=0.65,
                        help="Minimum fraction of consistency passes that must agree on the predicted class.")
    parser.add_argument("--self-train-consistency-power", type=float, default=1.0,
                        help="Exponent applied when translating consistency scores into pseudo-label weights (>=0).")
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
    parser.add_argument("--class-balance-strategy",
                        choices=["none", "recall", "f1", "precision_recall"],
                        default="recall",
                        help="Strategy used to derive dynamic class weighting from validation metrics.")
    parser.add_argument("--class-balance-boost", type=float, default=1.4,
                        help="Scaling factor applied to the class-balance error signal (higher emphasises hard classes).")
    parser.add_argument("--class-balance-power", type=float, default=1.5,
                        help="Exponent applied to the class-balance error signal before scaling (>=1).")
    parser.add_argument("--class-balance-momentum", type=float, default=0.85,
                        help="Momentum used when smoothing successive class-balance multipliers (0 disables smoothing).")
    parser.add_argument("--class-balance-min-multiplier", type=float, default=0.5,
                        help="Lower bound applied to class-specific multipliers when reweighting samples.")
    parser.add_argument("--class-balance-max-multiplier", type=float, default=3.5,
                        help="Upper bound applied to class-specific multipliers when reweighting samples.")
    parser.add_argument("--class-balance-floor", type=float, default=0.75,
                        help="Baseline multiplier floor that prevents classes from being suppressed below this value.")
    parser.add_argument("--class-balance-min-support", type=int, default=12,
                        help="Minimum validation support required before a class can receive the full reweighting signal.")
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
    parser.add_argument("--metadata-feature-strategy",
                        choices=["none", "one_hot"],
                        default="one_hot",
                        help="Strategy used to transform structured metadata columns into auxiliary features.")
    parser.add_argument("--metadata-min-frequency", type=int, default=3,
                        help="Minimum count required before a metadata value is assigned a dedicated feature column.")
    parser.add_argument("--metadata-include-missing", dest="metadata_include_missing", action="store_true",
                        help="Add an explicit slot representing missing or unseen metadata values when encoding features.")
    parser.add_argument("--metadata-skip-missing", dest="metadata_include_missing", action="store_false",
                        help="Do not add a dedicated slot for missing metadata values when encoding features.")
    parser.set_defaults(metadata_include_missing=True)
    parser.add_argument("--keyword-calibration", dest="keyword_calibration", action="store_true",
                        help="Enable lexical keyword priors that adjust logits during evaluation and pseudo-labelling.")
    parser.add_argument("--no-keyword-calibration", dest="keyword_calibration", action="store_false",
                        help="Disable keyword-based calibration heuristics.")
    parser.set_defaults(keyword_calibration=True)
    parser.add_argument("--keyword-calibration-min-frequency", type=int, default=3,
                        help="Minimum document frequency required before a unigram contributes to keyword calibration.")
    parser.add_argument("--keyword-calibration-bigram-min-frequency", type=int, default=2,
                        help="Minimum document frequency required before a bigram contributes to keyword calibration.")
    parser.add_argument("--keyword-calibration-smoothing", type=float, default=0.35,
                        help="Additive smoothing applied when estimating keyword-conditioned label probabilities (>=0).")
    parser.add_argument("--keyword-calibration-strength-threshold", type=float, default=0.1,
                        help="Minimum absolute log-odds strength required for a keyword feature to be retained.")
    parser.add_argument("--keyword-calibration-max-features", type=int, default=40,
                        help="Maximum number of keyword features preserved per label during calibration fitting.")
    parser.add_argument("--keyword-calibration-bias-weight", type=float, default=0.75,
                        help="Scale applied to class-prior logits contributed by the keyword calibrator (>=0).")
    parser.add_argument("--keyword-calibration-feature-weight", type=float, default=1.1,
                        help="Scale applied to keyword feature logits before fusing with model predictions (>=0).")
    parser.add_argument("--keyword-calibration-normalise-power", type=float, default=0.5,
                        help="Exponent used when normalising multiple keyword matches inside one example (>=0).")
    parser.add_argument("--meta-stacker", dest="meta_stacker", action="store_true",
                        help="Enable the stacked logistic meta-learner that refines logits using auxiliary features.")
    parser.add_argument("--no-meta-stacker", dest="meta_stacker", action="store_false",
                        help="Disable the stacked logistic meta-learner for logit refinement.")
    parser.set_defaults(meta_stacker=False)
    parser.add_argument("--meta-stacker-scale", type=float, default=0.85,
                        help="Scaling factor applied to meta-stacker outputs before they are fused with logits (>=0).")
    parser.add_argument("--meta-stacker-regularization", type=float, default=4.0,
                        help="Inverse regularisation strength (C) used by the meta-stacker logistic regression (>0).")
    parser.add_argument("--meta-stacker-max-iter", type=int, default=500,
                        help="Maximum optimisation iterations allowed when fitting the meta-stacker (>=50).")
    parser.add_argument("--meta-stacker-min-accuracy", type=float, default=0.55,
                        help="Minimum training accuracy required for the meta-stacker to remain enabled (0-1).")
    parser.add_argument("--cognitive-router", dest="cognitive_router", action="store_true",
                        help="Enable the cognitive-symbolic router that injects rule-based logit adjustments.")
    parser.add_argument("--no-cognitive-router", dest="cognitive_router", action="store_false",
                        help="Disable the cognitive-symbolic router adjustments.")
    parser.set_defaults(cognitive_router=True)
    parser.add_argument("--cognitive-router-signal-scale", type=float, default=2.6,
                        help="Base magnitude applied to positive router triggers when boosting intent logits (>=0).")
    parser.add_argument("--cognitive-router-penalty-scale", type=float, default=1.8,
                        help="Magnitude applied when the router downranks conflicting intents (>=0).")
    parser.add_argument("--cognitive-router-synergy-scale", type=float, default=0.75,
                        help="Additional reinforcement added for subsequent router triggers within one example (>=0).")
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
    parser.add_argument("--fp16", dest="fp16", action="store_true",
                        help="Enable mixed-precision training when CUDA/AMP are available.")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="Disable mixed-precision training even when CUDA is available.")
    parser.set_defaults(fp16=True)
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Compute device preference (auto, cpu, cuda, cuda:<index>, or mps). "
            "When CUDA is unavailable the trainer now continues on CPU automatically."
        ),
    )
    parser.add_argument(
        "--cuda-min-free-gb",
        type=float,
        default=3.0,
        help=(
            "Minimum free GPU memory (GiB) required to keep CUDA active. "
            "When the selected device reports less free VRAM the trainer falls back to CPU "
            "execution unless --require-cuda is supplied. Set to 0 to disable this safeguard."
        ),
    )
    parser.add_argument(
        "--cuda-max-utilization",
        type=float,
        default=0.95,
        help=(
            "Maximum VRAM utilisation ratio permitted before the micro-batch regulator shrinks batches. "
            "Values should fall in (0,1]; higher values allow fuller GPU usage before the guard activates."
        ),
    )
    parser.add_argument(
        "--cuda-sweep-reserved-gb",
        type=float,
        default=1.25,
        help=(
            "Reserved CUDA memory threshold (GiB) that triggers cache sweeping. "
            "Set to 0 to disable reserved-memory-driven sweeps."
        ),
    )
    parser.add_argument(
        "--cuda-sweep-free-gb",
        type=float,
        default=2.5,
        help=(
            "Trigger a CUDA cache sweep when free VRAM falls at or below this value (GiB). "
            "Set to 0 to disable free-memory-driven sweeps."
        ),
    )
    parser.add_argument(
        "--cuda-sweep-interval",
        type=int,
        default=32,
        help=(
            "Minimum number of micro-batch slices between CUDA cache sweeps (>=1)."
        ),
    )
    parser.add_argument(
        "--allow-cpu-testing",
        action="store_true",
        help=(
            "Preserved for backwards compatibility; CPU execution is now enabled by default."
        ),
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help=(
            "Abort if CUDA acceleration cannot be activated. When the CUDA-enabled PyTorch wheel "
            "is missing the trainer will attempt to install it automatically."
        ),
    )
    parser.add_argument(
        "--verify-device-only",
        action="store_true",
        help=(
            "Stop after validating the compute device configuration. "
            "Use together with --epochs 0 for lightweight environment checks."
        ),
    )
    parser.add_argument("--dataloader-workers", type=int, default=None,
                        help=(
                            "Number of CPU worker processes dedicated to background data preloading. "
                            "Defaults to an auto-selected value when using CUDA."
                        ))
    parser.add_argument("--dataloader-prefetch", type=int, default=2,
                        help=(
                            "Number of batches each CPU worker prefetches when CUDA is active (ignored when workers=0)."
                        ))
    parser.add_argument(
        "--cuda-prefetch-prime",
        dest="cuda_prefetch_prime",
        action="store_true",
        default=None,
        help=(
            "Synchronise the initial CUDA prefetch queue so device memory usage reaches steady-state immediately."
        ),
    )
    parser.add_argument(
        "--no-cuda-prefetch-prime",
        dest="cuda_prefetch_prime",
        action="store_false",
        help="Disable CUDA prefetch priming even when performance overdrive is active.",
    )
    parser.add_argument("--memory-guard", dest="memory_guard", action="store_true",
                        help="Enable automatic parameter reductions when host RAM is limited.")
    parser.add_argument("--no-memory-guard", dest="memory_guard", action="store_false",
                        help="Disable automatic memory guard heuristics (default).")
    parser.set_defaults(memory_guard=False)
    parser.add_argument("--performance-overdrive", dest="performance_overdrive", action="store_true",
                        help="Drive the trainer to saturate CPU/GPU resources for maximum throughput (default).")
    parser.add_argument("--no-performance-overdrive", dest="performance_overdrive", action="store_false",
                        help="Disable the aggressive performance overdrive heuristics.")
    parser.set_defaults(performance_overdrive=True)
    parser.add_argument("--hardware-monitor-interval", type=float, default=0.5,
                        help="Polling interval (in seconds) for hardware telemetry sampling (must be > 0).")
    parser.add_argument(
        "--speed-test",
        action="store_true",
        help="Emit dataset loading and training throughput statistics at the end of the run.",
    )
    parser.add_argument(
        "--estimate-dataset",
        type=Path,
        default=Path("data/intent_dataset.csv"),
        help="Dataset path used when projecting runtime for the full training corpus.",
    )
    parser.add_argument(
        "--estimate-dataset-scan-limit",
        type=int,
        default=4096,
        help=(
            "Maximum labelled examples sampled when measuring token statistics for runtime projections."
        ),
    )
    parser.add_argument(
        "--speed-test-reference-gflops",
        type=float,
        default=0.0,
        help="Override the sustained GFLOP/s throughput used for runtime projections (GFLOP/s).",
    )
    parser.add_argument(
        "--speed-test-calibration-seconds",
        type=float,
        default=0.0,
        help="Observed wall-clock seconds for a calibration epoch on the target dataset.",
    )
    parser.add_argument(
        "--speed-test-calibration-examples",
        type=int,
        default=0,
        help="Number of labelled examples processed during the calibration measurement.",
    )
    parser.add_argument(
        "--speed-test-calibration-epochs",
        type=float,
        default=1.0,
        help="Number of epochs completed during the calibration measurement (default: 1).",
    )
    parser.add_argument("--nvidia-smi-binary", type=str, default="nvidia-smi",
                        help="Path to the nvidia-smi binary used when capturing GPU telemetry.")
    parser.add_argument("--overdrive-simulate", dest="overdrive_simulate", action="store_true",
                        help="Run GPU-bound warm-up simulations before training to validate overdrive configuration.")
    parser.add_argument("--skip-overdrive-simulations", dest="overdrive_simulate", action="store_false",
                        help="Skip the GPU warm-up simulations even when performance overdrive is enabled.")
    parser.set_defaults(overdrive_simulate=True)
    parser.add_argument("--overdrive-simulation-rounds", type=int, default=3,
                        help="Number of warm-up simulation rounds executed before training (>=0).")
    parser.add_argument("--overdrive-simulation-matrix", type=int, default=3072,
                        help="Square matrix dimension used inside each simulation round (>=128).")
    parser.add_argument("--overdrive-simulation-batch", type=int, default=2,
                        help="Batch dimension used for batched matrix multiplications during simulations (>=1).")
    parser.add_argument("--memory-budget-gb", type=float, default=0.0,
                        help="Trigger memory guard when total RAM is at or below this many GiB (0 uses an auto threshold).")
    parser.add_argument("--memory-guard-min-available-gb", type=float, default=0.0,
                        help="Trigger memory guard when available RAM drops below this many GiB (0 uses an auto threshold).")
    parser.add_argument("--strict-device", action="store_true",
                        help="Deprecated; GPU execution is now mandatory and enforced by default.")
    parser.add_argument("--auto-optimizations", dest="auto_optimizations", action="store_true",
                        help="Enable automatic GPU-aware training optimisations (enabled by default).")
    parser.add_argument("--no-auto-optimizations", dest="auto_optimizations", action="store_false",
                        help="Disable automatic GPU-aware training optimisations.")
    parser.set_defaults(auto_optimizations=True)
    parser.add_argument("--overdrive-profile", action="store_true",
                        help="Enable high-capacity defaults that aggressively scale network width and training depth.")
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
    raw_argv = sys.argv[1:]
    parsed_argv, ignored_kernel_args = _strip_ipykernel_arguments(raw_argv)
    args, unknown_args = parser.parse_known_args(parsed_argv)
    if ignored_kernel_args:
        print(
            "Ignoring IPython kernel launcher arguments:",
            " ".join(ignored_kernel_args),
        )
    if unknown_args:
        parser.error("unrecognized arguments: " + " ".join(unknown_args))

    cuda_prefetch_prime_override = args.cuda_prefetch_prime is not None
    if args.cuda_prefetch_prime is None:
        args.cuda_prefetch_prime = False
    args.cuda_prefetch_prime_user_override = cuda_prefetch_prime_override

    if not numpy_available and not (args.verify_device_only or args.allow_cpu_testing):
        parser.error(
            "NumPy is required for intent classifier training. Install it with 'pip install numpy' "
            "or supply --verify-device-only/--allow-cpu-testing for diagnostic runs."
        )

    args.estimate_dataset = args.estimate_dataset.expanduser()

    model_dir_user_override = False
    for token in parsed_argv:
        if token == "--model-dir":
            model_dir_user_override = True
            break
        if token.startswith("--model-dir="):
            model_dir_user_override = True
            break

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
    elif args.self_train_rounds > 0:
        default_unlabeled = Path("data/unlabeled_pool.csv")
        try:
            args.unlabeled_dataset = resolve_training_input_path(
                default_unlabeled,
                description="unlabeled dataset",
                flag="--unlabeled-dataset",
                search_names=("unlabeled_pool.csv",),
            )
            print(f"Auto-detected unlabeled dataset at {args.unlabeled_dataset}")
        except FileNotFoundError:
            print(
                "Warning: self-training requested but no unlabeled dataset was found. "
                "Provide --unlabeled-dataset to enable pseudo-labelling."
            )

    _apply_colab_environment_adjustments(args, model_dir_user_override=model_dir_user_override)
    args.model_dir = Path(args.model_dir).expanduser()

    resolved_encoder, encoder_warning = _resolve_encoder_choice(args.encoder_type)
    if encoder_warning and resolved_encoder == args.encoder_type:
        print(f"WARNING: {encoder_warning}.")
    elif resolved_encoder != args.encoder_type:
        note = encoder_warning or "dependency constraint"
        print(
            f"WARNING: Encoder '{args.encoder_type}' is unavailable ({note}). "
            f"Falling back to '{resolved_encoder}'."
        )
        args.encoder_type = resolved_encoder

    device_pref_raw = str(args.device or "auto").strip().lower()

    if (
        device_pref_raw in {"", "auto"}
        and not args.require_cuda
        and not args.allow_cpu_testing
    ):
        cuda_status = _probe_torch_cuda_status()
        if cuda_status and cuda_status.get("runtime_available"):
            args.require_cuda = True
            os.environ[_REQUIRE_CUDA_ENV] = "1"
            print(
                "CUDA runtime detected; enforcing GPU execution. "
                "Pass --allow-cpu-testing to permit CPU fallback."
            )

    if args.verify_device_only and args.epochs != 0:
        parser.error("--verify-device-only must be combined with --epochs 0.")

    if args.st_hidden_dim < 1:
        parser.error("--st-hidden-dim must be positive.")
    if args.st_mlp_layers < 1:
        parser.error("--st-mlp-layers must be at least 1.")
    if args.st_mlp_expansion < 1.0:
        parser.error("--st-mlp-expansion must be >= 1.0.")
    if not 0 <= args.st_final_dropout < 1:
        parser.error("--st-final-dropout must lie in [0, 1).")
    if args.st_moe_experts < 0:
        parser.error("--st-moe-experts must be non-negative.")
    if 0 < args.st_moe_experts < 2:
        parser.error("--st-moe-experts must be at least 2 when enabling the mixture head.")
    if args.st_moe_hidden_dim < 0:
        parser.error("--st-moe-hidden-dim must be non-negative.")
    if not 0 <= args.st_moe_dropout < 1:
        parser.error("--st-moe-dropout must lie in [0, 1).")
    if args.vocab_char_ngram_min < 1:
        parser.error("--vocab-char-ngram-min must be at least 1.")
    if args.vocab_char_ngram_max < args.vocab_char_ngram_min:
        parser.error("--vocab-char-ngram-max must be greater than or equal to --vocab-char-ngram-min.")
    if args.vocab_char_ngram_limit < 0:
        parser.error("--vocab-char-ngram-limit must be non-negative.")
    if args.vocab_max_size < 0:
        parser.error("--vocab-max-size must be non-negative.")
    if args.min_freq < 1:
        parser.error("--min-freq must be at least 1.")
    if args.bilstm_conv_channels <= 0:
        parser.error("--bilstm-conv-channels must be positive.")
    if not 0 <= args.bilstm_conv_dropout < 1:
        parser.error("--bilstm-conv-dropout must lie in [0, 1).")
    if args.st_moe_temperature <= 0:
        parser.error("--st-moe-temperature must be positive.")
    if args.st_moe_topk < 0:
        parser.error("--st-moe-topk must be non-negative.")
    if args.st_moe_experts >= 2 and args.st_moe_topk > args.st_moe_experts:
        parser.error("--st-moe-topk cannot exceed --st-moe-experts when the mixture is enabled.")
    if args.st_moe_entropy_weight < 0:
        parser.error("--st-moe-entropy-weight must be non-negative.")
    if args.st_moe_balance_weight < 0:
        parser.error("--st-moe-balance-weight must be non-negative.")
    if not 0 <= args.st_moe_utilisation_momentum < 1:
        parser.error("--st-moe-utilisation-momentum must lie in [0, 1).")

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
    if args.self_train_consistency_passes < 1:
        parser.error("--self-train-consistency-passes must be at least 1.")
    if args.self_train_consistency_max_std < 0:
        parser.error("--self-train-consistency-max-std must be non-negative.")
    if not 0 <= args.self_train_consistency_min_agreement <= 1:
        parser.error("--self-train-consistency-min-agreement must lie in [0, 1].")
    if args.self_train_consistency_power < 0:
        parser.error("--self-train-consistency-power must be non-negative.")
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
    if args.micro_batch_size < 0:
        parser.error("--micro-batch-size must be non-negative.")
    if not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must lie in [0, 1).")
    if args.ema_start_epoch < 1:
        parser.error("--ema-start-epoch must be at least 1.")
    if args.cuda_sweep_interval < 1:
        parser.error("--cuda-sweep-interval must be at least 1.")
    if args.cuda_sweep_reserved_gb < 0:
        parser.error("--cuda-sweep-reserved-gb must be non-negative.")
    if args.cuda_sweep_free_gb < 0:
        parser.error("--cuda-sweep-free-gb must be non-negative.")
    if args.swa_start_epoch < 0:
        parser.error("--swa-start-epoch must be non-negative.")
    if args.swa_lr < 0:
        parser.error("--swa-lr must be non-negative.")
    if args.swa_anneal_epochs < 1:
        parser.error("--swa-anneal-epochs must be at least 1.")
    dtype_aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "f32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "f16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtype_key = str(args.ema_offload_dtype).lower().strip()
    if dtype_key not in dtype_aliases:
        parser.error("--ema-offload-dtype must be one of float32, float16, bfloat16.")
    args.ema_offload_dtype_resolved = dtype_aliases[dtype_key]
    args.ema_offload_dtype = dtype_key
    if not 0 <= args.augment_probability <= 1:
        parser.error("--augment-probability must lie in [0, 1].")
    if args.distill_epochs < 0:
        parser.error("--distill-epochs must be non-negative.")
    if not 0 <= args.distill_alpha <= 1:
        parser.error("--distill-alpha must lie in [0, 1].")
    if args.distill_temperature <= 0:
        parser.error("--distill-temperature must be positive.")
    if args.hardware_monitor_interval <= 0:
        parser.error("--hardware-monitor-interval must be positive.")
    if args.overdrive_simulation_rounds < 0:
        parser.error("--overdrive-simulation-rounds must be non-negative.")
    if args.overdrive_simulation_matrix < 128:
        parser.error("--overdrive-simulation-matrix must be at least 128.")
    if args.overdrive_simulation_batch < 1:
        parser.error("--overdrive-simulation-batch must be at least 1.")
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
    if args.class_balance_boost < 0:
        parser.error("--class-balance-boost must be non-negative.")
    if args.class_balance_power <= 0:
        parser.error("--class-balance-power must be strictly positive.")
    if not 0 <= args.class_balance_momentum < 1:
        parser.error("--class-balance-momentum must lie in [0, 1).")
    if args.class_balance_min_multiplier <= 0:
        parser.error("--class-balance-min-multiplier must be positive.")
    if args.class_balance_max_multiplier < args.class_balance_min_multiplier:
        parser.error("--class-balance-max-multiplier must be >= --class-balance-min-multiplier.")
    if args.class_balance_floor <= 0:
        parser.error("--class-balance-floor must be strictly positive.")
    if args.class_balance_floor > args.class_balance_max_multiplier:
        parser.error("--class-balance-floor must be <= --class-balance-max-multiplier.")
    if args.class_balance_min_support < 0:
        parser.error("--class-balance-min-support must be non-negative.")
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
    if args.metadata_min_frequency < 1:
        parser.error("--metadata-min-frequency must be at least 1.")
    if args.keyword_calibration_min_frequency < 1:
        parser.error("--keyword-calibration-min-frequency must be at least 1.")
    if args.keyword_calibration_bigram_min_frequency < 1:
        parser.error("--keyword-calibration-bigram-min-frequency must be at least 1.")
    if args.keyword_calibration_smoothing < 0:
        parser.error("--keyword-calibration-smoothing must be non-negative.")
    if args.keyword_calibration_strength_threshold < 0:
        parser.error("--keyword-calibration-strength-threshold must be non-negative.")
    if args.keyword_calibration_max_features < 1:
        parser.error("--keyword-calibration-max-features must be at least 1.")
    if args.keyword_calibration_bias_weight < 0:
        parser.error("--keyword-calibration-bias-weight must be non-negative.")
    if args.keyword_calibration_feature_weight < 0:
        parser.error("--keyword-calibration-feature-weight must be non-negative.")
    if args.keyword_calibration_normalise_power < 0:
        parser.error("--keyword-calibration-normalise-power must be non-negative.")
    if args.meta_stacker_scale < 0:
        parser.error("--meta-stacker-scale must be non-negative.")
    if args.meta_stacker_regularization <= 0:
        parser.error("--meta-stacker-regularization must be positive.")
    if args.meta_stacker_max_iter < 50:
        parser.error("--meta-stacker-max-iter must be at least 50.")
    if not 0 <= args.meta_stacker_min_accuracy <= 1:
        parser.error("--meta-stacker-min-accuracy must lie in [0, 1].")
    if args.cognitive_router_signal_scale < 0:
        parser.error("--cognitive-router-signal-scale must be non-negative.")
    if args.cognitive_router_penalty_scale < 0:
        parser.error("--cognitive-router-penalty-scale must be non-negative.")
    if args.cognitive_router_synergy_scale < 0:
        parser.error("--cognitive-router-synergy-scale must be non-negative.")

    default_transformer_model = parser.get_default("transformer_model")
    default_class_balance_strategy = parser.get_default("class_balance_strategy")

    if args.overdrive_profile:
        overdrive_changes: list[str] = []

        def _ensure_min(name: str, target: int | float) -> None:
            current = getattr(args, name)
            if current < target:
                setattr(args, name, target)
                overdrive_changes.append(f"{name}: {current} -> {target}")

        def _ensure_value(name: str, target: int | float | str | bool) -> None:
            current = getattr(args, name)
            if current != target:
                setattr(args, name, target)
                overdrive_changes.append(f"{name}: {current} -> {target}")

        def _ensure_flag(name: str) -> None:
            if not getattr(args, name):
                setattr(args, name, True)
                overdrive_changes.append(f"{name}: False -> True")

        _ensure_min("epochs", 48)
        _ensure_min("batch_size", 48)
        _ensure_min("grad_accumulation_steps", 4)
        _ensure_min("self_train_rounds", 3)
        _ensure_min("self_train_epochs", 3)
        _ensure_min("self_play_rounds", 2)
        _ensure_min("self_play_epochs", 2)
        _ensure_min("self_play_per_label", 5)
        _ensure_min("self_play_samples", 6)
        _ensure_min("self_train_consistency_passes", 6)
        _ensure_min("self_train_consistency_min_agreement", 0.7)
        _ensure_min("self_train_weight", 0.75)
        _ensure_min("self_train_confidence_power", 1.25)
        _ensure_min("distill_epochs", 2)
        _ensure_min("final_train_epochs", 4)
        if args.final_train_learning_rate == 0:
            _ensure_value("final_train_learning_rate", args.learning_rate * 0.6)
        if args.final_train_weight_decay < 0:
            _ensure_value("final_train_weight_decay", args.weight_decay)
        if args.final_train_batch_size == 0:
            _ensure_value("final_train_batch_size", max(args.batch_size, 48))

        if args.ema_decay > 0:
            _ensure_value("ema_decay", 0.0)
        if getattr(args, "ema_use_for_eval", False):
            args.ema_use_for_eval = False
            overdrive_changes.append("ema_use_for_eval: True -> False")

        swa_window = max(1, int(round(args.epochs * 0.15)))
        desired_swa_start = max(args.epochs - swa_window, 1)
        _ensure_value("swa_start_epoch", desired_swa_start)
        if args.swa_lr <= 0 or args.swa_lr > args.learning_rate:
            _ensure_value("swa_lr", max(args.learning_rate * 0.3, 5e-6))

        if args.scheduler == "none":
            _ensure_value("scheduler", "cosine")
        if args.class_balance_strategy == default_class_balance_strategy:
            _ensure_value("class_balance_strategy", "precision_recall")

        if args.encoder_type == "transformer" and args.transformer_model == default_transformer_model:
            _ensure_value("transformer_model", "bert-base-uncased")
            _ensure_value("transformer_learning_rate", min(args.transformer_learning_rate, 5e-5))

        if args.encoder_type == "bilstm":
            _ensure_min("embedding_dim", 256)
            _ensure_min("hidden_dim", 384)
            _ensure_min("ffn_dim", 1024)
            _ensure_min("encoder_layers", 3)
            _ensure_min("attention_heads", 8)
            _ensure_min("dropout", 0.35)
            _ensure_min("rdrop_alpha", 0.25)
            _ensure_flag("bilstm_conv_head")
            _ensure_min("bilstm_conv_channels", 256)
            _ensure_min("bilstm_conv_dropout", 0.25)
        elif args.encoder_type == "st":
            _ensure_min("st_hidden_dim", 1024)
            _ensure_min("st_dropout", 0.25)
            _ensure_min("st_mlp_layers", 4)
            _ensure_min("st_mlp_expansion", 1.6)
            _ensure_min("st_final_dropout", 0.1)
            _ensure_flag("st_mlp_layer_norm")
            _ensure_flag("st_mlp_residual")
            _ensure_min("st_moe_experts", 6)
            if args.st_moe_hidden_dim <= 0:
                _ensure_value("st_moe_hidden_dim", max(args.st_hidden_dim * 2, 1024))
            _ensure_min("st_moe_dropout", 0.15)
            _ensure_min("st_moe_entropy_weight", 0.08)
            _ensure_min("st_moe_balance_weight", 0.08)
            if args.st_moe_topk == 0:
                _ensure_value("st_moe_topk", min(4, max(2, args.st_moe_experts)))
            _ensure_value("st_moe_activation", "gelu")
            _ensure_flag("st_moe_layer_norm")
        else:
            _ensure_min("learning_rate", max(args.learning_rate, 2e-4))

        _ensure_flag("enable_emotion_reasoner")
        _ensure_flag("meta_introspector")
        _ensure_flag("neuro_symbolic_reasoner")
        _ensure_flag("self_discovery")
        _ensure_flag("transcendent_cognition")
        _ensure_flag("frontier_intelligence")

        if overdrive_changes:
            print("Overdrive profile adjustments:")
            for change in overdrive_changes:
                print(f"  - {change}")
        else:
            print("Overdrive profile requested; existing arguments already satisfy high-capacity thresholds.")

    if args.st_moe_experts < 2:
        args.st_moe_topk = 0
    else:
        args.st_moe_topk = min(args.st_moe_topk, args.st_moe_experts)
    args.st_moe_enabled = bool(args.st_moe_experts >= 2)
    args.st_moe_effective_hidden_dim = (
        args.st_moe_hidden_dim if args.st_moe_hidden_dim > 0 else args.st_hidden_dim
    ) if args.st_moe_enabled else 0

    st_hidden_dims = progressive_mlp_hidden_dims(
        args.st_hidden_dim,
        args.st_mlp_layers,
        args.st_mlp_expansion,
    )
    args.st_mlp_hidden_dims = st_hidden_dims

    augment_strategies = [strategy.strip() for strategy in args.augment_strategies.split(",") if strategy.strip()]
    if args.augment_probability > 0 and not augment_strategies:
        parser.error("--augment-probability requires at least one augmentation strategy.")

    _apply_memory_guard(args)

    set_seed(args.seed)

    speed_logger = SpeedTestLogger(args.speed_test)

    if not args.dataset.exists():
        raise FileNotFoundError(f"Labelled dataset not found: {args.dataset}")

    dataset_checksum = compute_sha1(args.dataset)
    dataset_timer = speed_logger.marker()
    texts, labels, metadata_rows = read_dataset(args.dataset)
    token_sample_limit = int(getattr(args, "estimate_dataset_scan_limit", 0) or 0)
    dataset_summary = _build_dataset_summary_from_texts(
        texts,
        path=args.dataset,
        sample_limit=token_sample_limit if token_sample_limit > 0 else None,
    )
    average_tokens = dataset_summary.average_tokens
    dataset_notes: dict[str, Any] = {
        "average_tokens": float(average_tokens),
        "token_samples": float(dataset_summary.token_samples),
        "estimated_tokens": float(dataset_summary.total_tokens),
    }
    if dataset_summary.examples > 0 and dataset_summary.token_samples > 0:
        dataset_notes["token_sample_fraction"] = (
            float(dataset_summary.token_samples) / float(dataset_summary.examples)
        )
    speed_logger.record_section(
        "dataset_load",
        dataset_timer,
        count=len(texts),
        notes=dataset_notes,
        add_to_total=False,
    )
    if not texts:
        raise RuntimeError(f"Dataset at {args.dataset} is empty.")

    raw_vocab_sources = [entry.strip().lower() for entry in args.vocab_extra_corpus.split(",") if entry.strip()]
    vocab_extra_sources = {source for source in raw_vocab_sources if source != "none"}

    unlabeled_master: list[str] = []
    unlabeled_checksum: str | None = None
    if args.unlabeled_dataset:
        if not args.unlabeled_dataset.exists():
            raise FileNotFoundError(f"Unlabeled dataset not found: {args.unlabeled_dataset}")
        unlabeled_master = read_unlabeled_dataset(args.unlabeled_dataset)
        unlabeled_checksum = compute_sha1(args.unlabeled_dataset)
        print(f"Loaded {len(unlabeled_master)} unlabeled examples for self-training.")
    elif args.self_train_rounds > 0:
        print("No unlabeled dataset supplied; skipping self-training despite configured rounds.")

    # Resolve the compute device early so preprocessing steps (tokenisation, embedding,
    # caching) can immediately leverage GPU acceleration when available.
    device, device_info = resolve_training_device(
        args.device,
        allow_fallback=not args.require_cuda,
    )
    device, device_info = _enforce_cuda_memory_budget(args, device, device_info)
    fallback_reason = cast(str | None, device_info.get("fallback"))
    available_backends = cast(dict[str, list[str]], device_info.get("available", {}))
    using_cuda = device.type == "cuda"
    using_mps = device.type == "mps"
    cuda_diagnostics: dict[str, Any] | None = None

    expect_cuda_flag = os.environ.pop(_EXPECT_CUDA_ENV, None)
    if expect_cuda_flag and not (using_cuda or using_mps):
        message = (
            "A CUDA-enabled PyTorch build was installed earlier for GPU training, "
            "but CUDA is still unavailable at runtime. "
            "Verify the NVIDIA driver installation and ensure the GPU is accessible."
        )
        if args.allow_cpu_testing:
            print(
                "Warning: "
                + message
                + " Continuing with CPU execution because --allow-cpu-testing was supplied."
            )
        else:
            raise RuntimeError(message + " Pass --allow-cpu-testing to override.")

    if using_cuda:
        amp_device_type = "cuda"
    elif using_mps:
        amp_device_type = "mps"
    else:
        amp_device_type = "cpu"

    vocab_config = VocabularyConfig(
        include_bigrams=args.vocab_include_bigrams,
        include_trigrams=args.vocab_include_trigrams,
        include_char_ngrams=args.vocab_include_char_ngrams,
        char_ngram_min=args.vocab_char_ngram_min,
        char_ngram_max=args.vocab_char_ngram_max,
        char_ngram_limit=args.vocab_char_ngram_limit,
    )

    vocab_extra_texts: list[str] = []
    active_vocab_sources: set[str] = set()
    if "unlabeled" in vocab_extra_sources and unlabeled_master:
        vocab_extra_texts.extend(unlabeled_master)
        active_vocab_sources.add("unlabeled")
    if "metadata" in vocab_extra_sources and metadata_rows:
        metadata_fragments = [value for row in metadata_rows for value in row.values() if value]
        if metadata_fragments:
            vocab_extra_texts.extend(metadata_fragments)
            active_vocab_sources.add("metadata")
    deduped_extra_texts: list[str] | None
    if vocab_extra_texts:
        deduped_extra_texts = list(dict.fromkeys(vocab_extra_texts))
        if active_vocab_sources:
            sources_str = ", ".join(sorted(active_vocab_sources))
            print(
                f"Vocabulary builder will incorporate {len(deduped_extra_texts)} extra text fragments from: {sources_str}."
            )
    else:
        deduped_extra_texts = None

    try:
        raw_conv_kernels = [
            int(value.strip())
            for value in args.bilstm_conv_kernels.split(",")
            if value.strip()
        ]
    except ValueError as exc:
        parser.error(f"--bilstm-conv-kernels must contain integers: {exc}")
        raw_conv_kernels = []  # pragma: no cover
    if any(kernel <= 0 for kernel in raw_conv_kernels):
        parser.error("--bilstm-conv-kernels values must be positive integers.")
    if not raw_conv_kernels and args.bilstm_conv_head:
        raw_conv_kernels = [3, 5, 7]
    bilstm_conv_kernel_sizes = raw_conv_kernels

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

    tokenizer_obj: TokenizerProtocol | None = None
    tokenizer_cache: TokenizerCacheFn | None = None
    embedding_fn: Callable[[str], VectorLike] | None = None
    embedding_batch_size = 0
    sentence_model = None
    sentence_embedding_dim: int | None = None
    embedding_cache_info: Callable[[], object] | None = None
    populate_sentence_cache_fn: Callable[[Sequence[str], str], None] | None = None
    populate_tokenizer_cache: Callable[[Sequence[str], str], None] | None = None
    dataset_embedding_target_device: TorchDevice | None = None
    sentence_embedding_cache_device: TorchDevice | None = None
    preferred_sentence_cache_device: TorchDevice | None = None
    sentence_embedding_cache_device_label = "cpu"
    estimated_sentence_cache_bytes: int | None = None
    if args.encoder_type == "transformer":
        tokenizer_obj = cast(TokenizerProtocol, load_transformer_tokenizer(args.transformer_model))
        max_seq_len = args.max_seq_len

        tokenizer_cache_store: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        chunk_size = max(32, args.batch_size * 4 if args.batch_size > 0 else 128)

        def _encode_batch(samples: Sequence[str]) -> None:
            if not samples:
                return
            tokenizer = tokenizer_obj
            if tokenizer is None:
                raise RuntimeError(
                    "Transformer tokenizer has not been initialised; cannot encode samples."
                )
            encoded = tokenizer(
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

        def populate_tokenizer_cache_impl(samples: Sequence[str], description: str) -> None:
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

        assert tokenizer_obj is not None

        def tokenizer_cache_lookup(sample: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
            cached = tokenizer_cache_store.get(sample)
            if cached is None:
                _encode_batch([sample])
                cached = tokenizer_cache_store[sample]
            return cached

        populate_tokenizer_cache = populate_tokenizer_cache_impl
        tokenizer_cache = tokenizer_cache_lookup
    elif args.encoder_type == "st":
        sentence_model = load_sentence_transformer(args.sentence_transformer_model)
        sentence_model = sentence_model.eval()
        if using_cuda or using_mps:
            target_device_str = str(device)
            try:
                sentence_model = sentence_model.to(device)
            except AttributeError:
                sentence_model = sentence_model.to(target_device_str)
        else:
            target_device_str = "cpu"
        if using_cuda or using_mps:
            preferred_sentence_cache_device = torch.device(
                device.type, device.index if device.index is not None else 0
            )
        else:
            preferred_sentence_cache_device = torch.device("cpu")
        sentence_embedding_cache_device = torch.device("cpu")
        sentence_embedding_cache_device_label = "cpu"
        dataset_embedding_target_device = None
        model_for_dimension = sentence_model
        if model_for_dimension is None:
            raise RuntimeError(
                "Sentence-transformer encoder failed to load; embedding dimension unknown."
            )
        sentence_embedding_dim = int(model_for_dimension.get_sentence_embedding_dimension())

        embedding_batch_size = max(32, min(1024, int(args.batch_size or DEFAULT_BATCH_SIZE) * 8))
        sentence_embedding_cache: dict[str, Any] = {}

        def _encode_sentence_batch(batch_samples: Sequence[str]) -> None:
            candidates = [sample for sample in batch_samples if sample and sample not in sentence_embedding_cache]
            if not candidates:
                return
            with torch.inference_mode():
                model_for_encoding = sentence_model
                if model_for_encoding is None:
                    raise RuntimeError(
                        "Sentence-transformer encoder unavailable during embedding batch execution."
                    )
                encoded = model_for_encoding.encode(
                    list(candidates),
                    batch_size=embedding_batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    device=target_device_str,
                )
            if torch.is_tensor(encoded):
                vectors = encoded.detach()
            else:
                vectors = torch.as_tensor(encoded)
            vectors = vectors.to(dtype=torch.float32)
            target_cache_device = sentence_embedding_cache_device or torch.device("cpu")
            target_device_type = target_cache_device.type
            if vectors.device != target_cache_device:
                non_blocking = target_device_type == "cuda"
                vectors = vectors.to(target_cache_device, non_blocking=non_blocking)
            if (
                target_device_type == "cpu"
                and using_cuda
                and hasattr(vectors, "is_pinned")
                and not vectors.is_pinned()
            ):
                vectors = vectors.pin_memory()
            for sample, vector in zip(candidates, vectors):
                if sample in sentence_embedding_cache:
                    continue
                sentence_embedding_cache[sample] = vector

        def populate_sentence_cache(samples: Sequence[str], description: str) -> None:
            unique_samples = [
                sample for sample in dict.fromkeys(samples)
                if sample and sample not in sentence_embedding_cache
            ]
            if not unique_samples:
                return
            print(
                f"Encoding {len(unique_samples)} texts on {target_device_str} for {description} "
                f"(batch size {embedding_batch_size}; cache {sentence_embedding_cache_device_label})."
            )
            for start in range(0, len(unique_samples), embedding_batch_size):
                batch = unique_samples[start:start + embedding_batch_size]
                _encode_sentence_batch(batch)

        def embedding_cache_info_fn() -> SimpleNamespace:
            device_str = (
                str(sentence_embedding_cache_device)
                if sentence_embedding_cache_device is not None
                else "cpu"
            )
            return SimpleNamespace(
                maxsize=0,
                currsize=len(sentence_embedding_cache),
                device=device_str,
                estimated_bytes=estimated_sentence_cache_bytes,
            )

        def embed_text(sample: str) -> TorchTensor:
            cached = sentence_embedding_cache.get(sample)
            if cached is None:
                _encode_sentence_batch([sample])
                cached = sentence_embedding_cache.get(sample)
                if cached is None:
                    raise RuntimeError(
                        "Sentence-transformer embedding cache did not capture an encoded sample; "
                        "ensure inputs are non-empty strings."
                    )
            return cached

        embedding_fn = embed_text
        populate_sentence_cache_fn = populate_sentence_cache
        max_seq_len = 1
        embedding_cache_info = embedding_cache_info_fn
    else:
        if texts:
            augmented_lengths = [len(generate_training_tokens(text, vocab_config)) for text in texts]
            max_tokens = max(augmented_lengths) if augmented_lengths else args.max_seq_len
        else:
            max_tokens = args.max_seq_len
        if args.auto_extend_max_seq and max_tokens > args.max_seq_len:
            headroom = max(1, int(math.ceil(max_tokens * 1.05)))
            if headroom != args.max_seq_len:
                print(
                    f"Auto-extending max sequence length from {args.max_seq_len} to {headroom} tokens to accommodate augmented vocabulary features."
                )
            max_seq_len = max(8, headroom)
        else:
            max_seq_len = max(8, min(args.max_seq_len, max_tokens))

    vocab = build_vocab(
        texts,
        min_freq=args.min_freq,
        config=vocab_config,
        extra_texts=deduped_extra_texts,
        max_size=args.vocab_max_size,
    )
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(label_to_idx)
    idx_to_label_list = ["?"] * num_classes
    for label, idx in label_to_idx.items():
        idx_to_label_list[idx] = label

    metadata_encoder: StructuredMetadataEncoder | None = None
    if args.metadata_feature_strategy != "none":
        metadata_encoder = StructuredMetadataEncoder(
            metadata_rows,
            min_frequency=args.metadata_min_frequency,
            include_missing=args.metadata_include_missing,
        )
        if metadata_encoder.dimension == 0:
            metadata_encoder = None

    lexicon_active = bool(args.enable_emotion_reasoner)
    emotion_lexicon = EmotionLexicon() if lexicon_active else None
    lexicon_dim = len(emotion_lexicon.emotions) if emotion_lexicon is not None else 0
    metadata_dim = metadata_encoder.dimension if metadata_encoder is not None else 0
    emotion_enabled = bool((lexicon_dim > 0) or (metadata_dim > 0))
    emotion_dim = lexicon_dim + metadata_dim

    args._preflight_vocab_size = len(vocab)
    args._preflight_max_seq_len = max_seq_len
    args._preflight_num_classes = num_classes
    args._preflight_emotion_dim = emotion_dim if emotion_enabled else 0
    if sentence_embedding_dim is not None:
        args._preflight_sentence_dim = sentence_embedding_dim
    else:
        args._preflight_sentence_dim = 0
    args._preflight_batch_size = int(max(2, min(getattr(args, "batch_size", 2) or 2, 8)))

    if fallback_reason:
        print(
            f"Device preference '{args.device}' could not be satisfied "
            f"({fallback_reason}); continuing with {device}."
        )
        lowered_pref = str(args.device or "").strip().lower()
        if lowered_pref and lowered_pref not in {"auto", "cpu"} and not using_cuda:
            _emit_cpu_bypass_diagnostics(device_info, fallback_reason)

    if using_cuda:
        device_index = device.index if device.index is not None else 0
        device_name = device_info.get("name") or f"cuda:{device_index}"
        if not torch.cuda.is_available():
            raise RuntimeError(
                "trainer_intent_classifier requires torch.cuda to be available after selecting a CUDA device."
            )
        torch.cuda.set_device(device)
        print(
            f"Using CUDA device {device_index} ({device_name}) for training; CPU workers will focus on preloading batches."
        )
        try:
            cuda_diagnostics = _gather_cuda_diagnostics(device)
        except Exception as diag_error:
            print(
                f"Warning: Unable to inspect CUDA device properties instantly ({diag_error}); proceeding without diagnostics."
            )
            cuda_diagnostics = None
        else:
            total_gib = float(cuda_diagnostics["total_memory_bytes"]) / (1024.0 ** 3)
            driver_version = cuda_diagnostics.get("driver_version")
            runtime_version = cuda_diagnostics.get("runtime_version")
            mp_count = cuda_diagnostics.get("multi_processor_count")
            capability = cuda_diagnostics.get("capability")
            print(
                "CUDA device ready: {name} | capability {capability} | "
                "{total_mem:.2f} GiB total VRAM | {mp_count} multiprocessors".format(
                    name=cuda_diagnostics.get("name", device_name),
                    capability=capability,
                    total_mem=total_gib,
                    mp_count=mp_count,
                )
            )
            if driver_version is not None or runtime_version is not None:
                driver_str = _format_cuda_driver_version(driver_version) or "unknown"
                runtime_str = "unknown" if runtime_version is None else str(runtime_version)
                print(
                    f"CUDA runtime version: {runtime_str} | NVIDIA driver version: {driver_str}"
                )
        print("Running a short CUDA warm-up to prime kernels and allocator.")
        _run_cuda_warmup(device)
    elif using_mps:
        device_index = None
        device_name = device_info.get("name") or "mps"
        print(f"Using Apple Metal (MPS) device {device_name} for training.")
    else:
        device_index = None
        device_name = str(device)
        if fallback_reason:
            print("Falling back to CPU execution; expect longer training times without GPU acceleration.")
        else:
            print("Using CPU device for training.")

    amp_available_global = bool(GradScaler is not None or using_mps)
    performance_overdrive_requested = bool(getattr(args, "performance_overdrive", True))
    performance_overdrive_active = performance_overdrive_requested and not getattr(
        args, "memory_guard_active", False
    )
    auto_actions = apply_auto_optimizations(
        args,
        dataset_size=len(texts),
        num_labels=num_classes,
        using_cuda=using_cuda,
        using_mps=using_mps,
        amp_available=amp_available_global,
        memory_guard=getattr(args, "memory_guard_active", False),
        performance_overdrive=performance_overdrive_active,
    )
    args.auto_optimizations_log = auto_actions

    _apply_performance_overdrive(
        args,
        using_cuda=using_cuda,
        using_mps=using_mps,
        dataset_size=len(texts),
        cuda_diagnostics=cuda_diagnostics if using_cuda else None,
        device=device if using_cuda else None,
    )

    micro_batch_size = _resolve_micro_batch_size(
        args,
        max(1, int(args.batch_size)),
        using_cuda=using_cuda,
        using_mps=using_mps,
    )
    args.micro_batch_size_effective = micro_batch_size
    vram_regulator: VramBudgetRegulator | None
    cuda_memory_sweeper: CudaMemorySweeper | None
    if using_cuda:
        regulator_min = max(8, micro_batch_size // 2)
        regulator_max = max(micro_batch_size, regulator_min)
        target_free = max(float(getattr(args, "cuda_min_free_gb", 0.0) or 0.0), 3.5)
        vram_regulator = VramBudgetRegulator(
            device=device,
            target_free_gb=target_free,
            max_utilization=float(args.cuda_max_utilization),
            min_micro_batch=regulator_min,
            max_micro_batch=regulator_max,
        )
        cuda_memory_sweeper = CudaMemorySweeper(
            device=device,
            trigger_reserved_gb=float(max(args.cuda_sweep_reserved_gb, 0.0)),
            trigger_free_gb=float(max(args.cuda_sweep_free_gb, 0.0)),
            interval=max(1, int(args.cuda_sweep_interval)),
        )
    else:
        vram_regulator = None
        cuda_memory_sweeper = None
    if micro_batch_size < int(args.batch_size):
        print(
            "Micro-batching: each loader batch of {base} samples will be processed as micro-batches of {micro}.".format(
                base=int(args.batch_size),
                micro=micro_batch_size,
            )
        )

    gpu_cached_embeddings = False
    dataset_pin_memory = using_cuda

    overdrive_active = bool(getattr(args, "performance_overdrive_active", False))
    if args.encoder_type == "st" and sentence_embedding_dim is not None:
        candidate_sequences: list[str] = list(texts)
        if unlabeled_master:
            candidate_sequences.extend(unlabeled_master)
        unique_candidates = [
            sample for sample in dict.fromkeys(candidate_sequences) if sample
        ]
        estimated_sentence_cache_bytes = (
            len(unique_candidates) * sentence_embedding_dim * 4
        )
        if unique_candidates:
            approx_gib = estimated_sentence_cache_bytes / float(1024 ** 3)
            print(
                f"Sentence-transformer cache will stage {len(unique_candidates)} unique texts "
                f"(~{approx_gib:.2f} GiB at dimension {sentence_embedding_dim})."
            )
        total_memory_bytes = 0
        if using_cuda:
            if cuda_diagnostics is not None:
                total_memory_bytes = int(cuda_diagnostics.get("total_memory_bytes", 0) or 0)
            if total_memory_bytes <= 0:
                try:
                    total_memory_bytes = int(torch.cuda.get_device_properties(device).total_memory)
                except Exception:
                    total_memory_bytes = 0
        if (
            using_cuda
            and preferred_sentence_cache_device is not None
            and total_memory_bytes > 0
            and estimated_sentence_cache_bytes > 0
        ):
            safe_budget = int(total_memory_bytes * 0.8)
            if estimated_sentence_cache_bytes <= safe_budget:
                cache_device = preferred_sentence_cache_device
                sentence_embedding_cache_device = cache_device
                sentence_embedding_cache_device_label = str(cache_device)
                dataset_embedding_target_device = cache_device
                gpu_cached_embeddings = cache_device.type == "cuda"
                print(
                    f"Storing sentence embeddings directly on {sentence_embedding_cache_device_label} "
                    "to avoid host/device copies."
                )
            else:
                sentence_embedding_cache_device = torch.device("cpu")
                sentence_embedding_cache_device_label = "cpu"
                dataset_embedding_target_device = None
                gpu_cached_embeddings = False
                oversub_ratio = estimated_sentence_cache_bytes / float(total_memory_bytes)
                print(
                    f"Keeping embeddings on CPU to respect VRAM budget ({oversub_ratio:.2%} of total memory)."
                )
        elif (
            preferred_sentence_cache_device is not None
            and preferred_sentence_cache_device.type == "mps"
        ):
            cache_device = preferred_sentence_cache_device
            sentence_embedding_cache_device = cache_device
            sentence_embedding_cache_device_label = str(cache_device)
            dataset_embedding_target_device = cache_device
        else:
            sentence_embedding_cache_device = torch.device("cpu")
            sentence_embedding_cache_device_label = "cpu"
            dataset_embedding_target_device = None
            gpu_cached_embeddings = False
        if using_cuda and total_memory_bytes > 0:
            tuned_batch = int(embedding_batch_size)
            total_gib = total_memory_bytes / float(1024 ** 3)
            if total_gib >= 60:
                tuned_batch = max(tuned_batch, 4096)
            elif total_gib >= 40:
                tuned_batch = max(tuned_batch, 3072)
            elif total_gib >= 24:
                tuned_batch = max(tuned_batch, 2048)
            max_candidate_batch = max(1, len(unique_candidates))
            tuned_batch = min(tuned_batch, max_candidate_batch, 8192)
            if tuned_batch != embedding_batch_size:
                embedding_batch_size = tuned_batch
                print(
                    f"Adjusted sentence embedding batch size to {embedding_batch_size} based on GPU VRAM ({total_gib:.1f} GiB)."
                )

        dataset_pin_memory = using_cuda and not gpu_cached_embeddings
    if using_cuda:
        auto_workers = _auto_dataloader_workers(
            performance_overdrive=overdrive_active
        )
        if args.dataloader_workers is None:
            dataloader_workers = max(1, auto_workers)
        else:
            dataloader_workers = max(1, int(args.dataloader_workers))
    else:
        if args.dataloader_workers is None:
            if using_mps:
                dataloader_workers = max(
                    1,
                    _auto_dataloader_workers(performance_overdrive=overdrive_active),
                )
            else:
                dataloader_workers = 0
        else:
            dataloader_workers = max(0, int(args.dataloader_workers))
    dataloader_prefetch = (
        max(1, int(args.dataloader_prefetch)) if dataloader_workers > 0 else 1
    )

    if using_cuda:
        print(
            f"Configuring {dataloader_workers} CPU worker(s) for asynchronous data preloading "
            f"(prefetch factor {dataloader_prefetch})."
        )
    elif using_mps:
        if dataloader_workers > 0:
            print(
                f"Configuring {dataloader_workers} worker(s) for the MPS backend "
                f"(prefetch factor {dataloader_prefetch})."
            )
        else:
            print("Data loading will proceed synchronously on the CPU (num_workers=0).")
    else:
        if dataloader_workers > 0:
            print(
                f"Configuring {dataloader_workers} CPU worker(s) for background data loading "
                f"(prefetch factor {dataloader_prefetch})."
            )
        else:
            print("Data loading will proceed synchronously on the CPU (num_workers=0).")

    if args.verify_device_only:
        if using_cuda:
            print(
                "CUDA device readiness confirmed instantly; exiting before training as requested."
            )
        else:
            print(
                f"Device verification completed on {device}; training was not executed."
            )
        return

    hardware_monitor = HardwareMonitorController(
        device.type,
        device_index=device_index if using_cuda else None,
        binary=args.nvidia_smi_binary,
        interval=args.hardware_monitor_interval,
    )
    if hardware_monitor.start():
        if using_cuda:
            display_index = device_index if device_index is not None else 0
            print(
                "Hardware telemetry: polling {binary} every {interval:.2f}s on GPU {index}.".format(
                    binary=args.nvidia_smi_binary,
                    interval=hardware_monitor.interval,
                    index=display_index,
                )
            )
        else:
            print(
                "Hardware telemetry: polling {binary} every {interval:.2f}s (backend active).".format(
                    binary=args.nvidia_smi_binary,
                    interval=hardware_monitor.interval,
                )
            )
    else:
        reason = hardware_monitor.reason or "unsupported-device"
        print(f"Hardware telemetry inactive ({reason}).")

    if getattr(args, "overdrive_simulate", False) and overdrive_active:
        overdrive_simulation_summary = _run_overdrive_simulations(
            args,
            device=device,
            using_cuda=using_cuda,
            using_mps=using_mps,
        )
    elif not getattr(args, "overdrive_simulate", False):
        overdrive_simulation_summary = {"enabled": False, "reason": "disabled"}
    else:
        overdrive_simulation_summary = {"enabled": False, "reason": "overdrive_inactive"}
    if not overdrive_simulation_summary.get("enabled"):
        print(
            f"Overdrive simulations skipped ({overdrive_simulation_summary.get('reason', 'disabled')})."
        )

    def create_data_loader(
        dataset: Dataset[BatchT],
        *,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoader[BatchT]:
        loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "num_workers": dataloader_workers,
            "pin_memory": dataset_pin_memory,
        }
        if dataloader_workers > 0:
            loader_kwargs["prefetch_factor"] = min(dataloader_prefetch, 2)
            loader_kwargs["persistent_workers"] = True
        if using_cuda and loader_kwargs.get("pin_memory"):
            gpu_index = device.index if device.index is not None else torch.cuda.current_device()
            loader_kwargs["pin_memory_device"] = f"cuda:{gpu_index}"
        try:
            base_loader: DataLoader[BatchT] = DataLoader(dataset, **loader_kwargs)
        except TypeError as exc:
            if "pin_memory_device" in str(exc):
                loader_kwargs.pop("pin_memory_device", None)
                base_loader = DataLoader(dataset, **loader_kwargs)
            else:
                raise
        return base_loader

    def build_model(target_device: TorchDevice | None = None) -> nn.Module:
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
            model_obj = model_obj.to(destination)
            return _finalise_model_for_training(model_obj, args, destination)
        if args.encoder_type == "st":
            if sentence_embedding_dim is None:
                raise RuntimeError("Sentence-transformer embedding dimension could not be determined.")
            model_obj = SentenceTransformerClassifier(
                embedding_dim=sentence_embedding_dim,
                hidden_dim=args.st_hidden_dim,
                num_classes=num_classes,
                dropout=args.st_dropout,
                num_layers=args.st_mlp_layers,
                expansion=args.st_mlp_expansion,
                activation=args.st_mlp_activation,
                final_dropout=args.st_final_dropout,
                use_layer_norm=args.st_mlp_layer_norm,
                use_residual=args.st_mlp_residual,
                moe_experts=args.st_moe_experts,
                moe_hidden_dim=args.st_moe_effective_hidden_dim,
                moe_activation=args.st_moe_activation,
                moe_dropout=args.st_moe_dropout,
                moe_temperature=args.st_moe_temperature,
                moe_topk=args.st_moe_topk,
                moe_entropy_weight=args.st_moe_entropy_weight,
                moe_balance_weight=args.st_moe_balance_weight,
                moe_use_layer_norm=args.st_moe_layer_norm,
                moe_utilisation_momentum=args.st_moe_utilisation_momentum,
            )
            if emotion_enabled and emotion_dim > 0:
                model_obj = EmotionallyAdaptiveModel(
                    model_obj,
                    num_classes=num_classes,
                    num_emotions=emotion_dim,
                    dropout=args.emotion_fusion_dropout,
                )
            model_obj = model_obj.to(destination)
            return _finalise_model_for_training(model_obj, args, destination)
        model_obj = IntentClassifier(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            num_layers=args.encoder_layers,
            attention_heads=args.attention_heads,
            ffn_dim=args.ffn_dim,
            use_conv_head=args.bilstm_conv_head,
            conv_kernel_sizes=bilstm_conv_kernel_sizes,
            conv_channels=args.bilstm_conv_channels,
            conv_dropout=args.bilstm_conv_dropout,
        )
        if emotion_enabled and emotion_dim > 0:
            model_obj = EmotionallyAdaptiveModel(
                model_obj,
                num_classes=num_classes,
                num_emotions=emotion_dim,
                dropout=args.emotion_fusion_dropout,
            )
        model_obj = model_obj.to(destination)
        return _finalise_model_for_training(model_obj, args, destination)

    unlabeled_master: list[str] = []
    unlabeled_checksum: str | None = None
    if args.unlabeled_dataset:
        if not args.unlabeled_dataset.exists():
            raise FileNotFoundError(f"Unlabeled dataset not found: {args.unlabeled_dataset}")
        unlabeled_master = read_unlabeled_dataset(args.unlabeled_dataset)
        unlabeled_checksum = compute_sha1(args.unlabeled_dataset)
        print(f"Loaded {len(unlabeled_master)} unlabeled examples for self-training.")
    elif args.self_train_rounds > 0:
        print("No unlabeled dataset supplied; skipping self-training despite configured rounds.")

    if args.encoder_type == "transformer" and populate_tokenizer_cache is not None:
        populate_tokenizer_cache(texts, "labelled dataset")
        if unlabeled_master:
            populate_tokenizer_cache(unlabeled_master, "unlabelled dataset")
    elif args.encoder_type == "st" and populate_sentence_cache_fn is not None:
        populate_sentence_cache_fn(texts, "labelled dataset embeddings")
        if unlabeled_master:
            populate_sentence_cache_fn(unlabeled_master, "unlabelled dataset embeddings")

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
    evaluation_inputs = build_advanced_valuation_suite()
    if (
        args.encoder_type == "transformer"
        and tokenizer_cache is not None
        and populate_tokenizer_cache is not None
    ):
        populate_tokenizer_cache(evaluation_inputs, "evaluation showcase set")
    elif args.encoder_type == "st" and populate_sentence_cache_fn is not None:
        populate_sentence_cache_fn(evaluation_inputs, "evaluation showcase embeddings")

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
        train_metadata = [metadata_rows[i] if i < len(metadata_rows) else None for i in train_indices]

        fold_keyword_calibrator: KeywordIntentCalibrator | None = None
        if args.keyword_calibration:
            fold_keyword_calibrator = build_keyword_intent_calibrator(
                train_texts,
                train_labels,
                label_to_idx=label_to_idx,
                min_frequency=args.keyword_calibration_min_frequency,
                bigram_min_frequency=args.keyword_calibration_bigram_min_frequency,
                smoothing=args.keyword_calibration_smoothing,
                strength_threshold=args.keyword_calibration_strength_threshold,
                max_features_per_label=args.keyword_calibration_max_features,
                bias_weight=args.keyword_calibration_bias_weight,
                feature_weight=args.keyword_calibration_feature_weight,
                normalise_power=args.keyword_calibration_normalise_power,
            )
            if fold_keyword_calibrator is None:
                print(
                    f"Fold {fold_index}/{total_folds}: keyword calibration disabled after fitting returned no usable features."
                )

        fold_cognitive_router: CognitiveIntentRouter | None = None
        if args.cognitive_router:
            fold_cognitive_router = CognitiveIntentRouter(
                label_to_idx=label_to_idx,
                signal_scale=args.cognitive_router_signal_scale,
                penalty_scale=args.cognitive_router_penalty_scale,
                synergy_scale=args.cognitive_router_synergy_scale,
            )

        fold_meta_stacker: MetaIntentStacker | None = None
        fold_meta_metadata: dict[str, Any] = {"enabled": bool(args.meta_stacker)}

        base_keyword_metadata = (
            fold_keyword_calibrator.export_metadata()
            if fold_keyword_calibrator is not None
            else {"enabled": bool(args.keyword_calibration)}
        )
        fold_keyword_metadata = _snapshot_mapping(base_keyword_metadata) or {
            "enabled": bool(args.keyword_calibration)
        }
        router_metadata_raw = (
            fold_cognitive_router.export_metadata()
            if fold_cognitive_router is not None
            else {"enabled": False}
        )
        router_metadata = _snapshot_mapping(router_metadata_raw) or {"enabled": False}
        fold_keyword_metadata["router"] = router_metadata
        fold_keyword_metadata["router_enabled"] = bool(router_metadata.get("enabled", False))
        fold_keyword_metadata["enabled"] = bool(
            fold_keyword_metadata.get("enabled", False) or router_metadata.get("enabled", False)
        )
        fold_keyword_metadata["router_trigger_total"] = int(
            router_metadata.get("positive_trigger_events", 0)
            + router_metadata.get("negative_trigger_events", 0)
        )

        train_emotion_vectors: list[list[float]] = []
        fold_emotion_memory: EmotionPrototypeMemory | None = None
        fold_emotion_config: EmotionTrainingConfig | None = None
        if emotion_enabled and emotion_dim > 0:
            train_emotion_vectors = [
                compose_emotion_features(
                    text,
                    train_metadata[idx] if idx < len(train_metadata) else None,
                    lexicon=emotion_lexicon if lexicon_active else None,
                    metadata_encoder=metadata_encoder,
                    lexicon_dim=lexicon_dim,
                    metadata_dim=metadata_dim,
                )
                for idx, text in enumerate(train_texts)
            ]
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

        fold_rdrop_config: RDropConfig | None = None
        if args.rdrop_alpha > 0:
            fold_rdrop_config = RDropConfig(
                enabled=True,
                alpha=float(args.rdrop_alpha),
                passes=max(2, int(args.rdrop_forward_passes)),
            )

        fold_meta_config: MetaCognitiveConfig | None = None
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

        fold_neuro_config: NeuroSymbolicConfig | None = None
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

        fold_discovery_config: SelfDiscoveryConfig | None = None
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

        fold_transcendent_config: TranscendentCognitionConfig | None = None
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

        fold_frontier_config: FrontierIntelligenceConfig | None = None
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

        class_balancer: ClassWeightBalancer | None = None
        if args.class_balance_strategy != "none":
            balance_config = ClassBalanceConfig(
                strategy=args.class_balance_strategy,
                boost=float(args.class_balance_boost),
                power=float(args.class_balance_power),
                momentum=float(args.class_balance_momentum),
                min_multiplier=float(args.class_balance_min_multiplier),
                max_multiplier=float(args.class_balance_max_multiplier),
                floor=float(args.class_balance_floor),
                min_support=int(args.class_balance_min_support),
            )
            class_balancer = ClassWeightBalancer(balance_config, train_labels)

        val_texts = [texts[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        if not val_texts:
            raise RuntimeError("Validation split produced no examples; adjust --test-ratio or --folds.")

        val_metadata: list[dict[str, str] | None] = [
            metadata_rows[i] if i < len(metadata_rows) else None for i in val_indices
        ]
        val_emotion_vectors: list[list[float]] | None = None
        if emotion_enabled and emotion_dim > 0:
            val_emotion_vectors = [
                compose_emotion_features(
                    text,
                    val_metadata[idx] if idx < len(val_metadata) else None,
                    lexicon=emotion_lexicon if lexicon_active else None,
                    metadata_encoder=metadata_encoder,
                    lexicon_dim=lexicon_dim,
                    metadata_dim=metadata_dim,
                )
                for idx, text in enumerate(val_texts)
            ]

        val_keyword_vectors: list[list[float]] | None = None
        if fold_keyword_calibrator is not None or fold_cognitive_router is not None:
            val_keyword_vectors = [
                compose_logit_adjustments(
                    text,
                    calibrator=fold_keyword_calibrator,
                    router=fold_cognitive_router,
                )
                for text in val_texts
            ]

        val_dataset = IntentDataset(
            val_texts,
            val_labels,
            vocab=vocab,
            vocab_config=vocab_config,
            label_to_idx=label_to_idx,
            max_len=max_seq_len,
            tokenizer=tokenizer_obj,
            tokenizer_cache=tokenizer_cache,
            embedding_model=embedding_fn,
            emotion_vectors=val_emotion_vectors,
            emotion_dim=emotion_dim if emotion_enabled else 0,
            keyword_vectors=val_keyword_vectors,
            pin_memory=dataset_pin_memory,
            target_device=dataset_embedding_target_device,
        )
        val_loader: DataLoader[BatchExample] = create_data_loader(
            val_dataset, batch_size=args.batch_size
        )

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
        if args.encoder_type == "transformer":
            optimizer = create_transformer_optimizer(
                model,
                base_lr=effective_lr,
                weight_decay=args.weight_decay,
                layerwise_decay=float(args.transformer_layerwise_decay),
            )
        else:
            optimizer = _memory_efficient_optimizer(
                model.parameters(),
                effective_lr,
                args.weight_decay,
            )

        amp_backend_available = bool(GradScaler is not None)
        if using_mps:
            amp_backend_available = amp_backend_available or _mps_backend_available()
        use_amp = bool(args.fp16 and (using_cuda or using_mps) and amp_backend_available)
        if args.fp16 and not (using_cuda or using_mps) and not fp16_warning_emitted:
            print(
                f"fp16 requested but the active device '{device.type}' does not support GPU/MPS AMP; "
                "training with full precision."
            )
            fp16_warning_emitted = True
        elif args.fp16 and (using_cuda or using_mps) and not amp_backend_available and not fp16_warning_emitted:
            print("fp16 requested but AMP utilities are unavailable on this device; training with full precision.")
            fp16_warning_emitted = True
        scaler = create_grad_scaler(use_amp and amp_device_type == "cuda", amp_device_type)

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
            lr_values = sorted({float(group.get("lr", effective_lr)) for group in optimizer.param_groups})
            if len(lr_values) > 1:
                print(
                    "   -> layer-wise learning rate span "
                    f"{lr_values[0]:.2e} – {lr_values[-1]:.2e} "
                    f"(decay {args.transformer_layerwise_decay:.3f})"
                )
        elif args.encoder_type == "st":
            cache_max = None
            primed = None
            if callable(embedding_cache_info):
                info = embedding_cache_info()
                cache_max = getattr(info, "maxsize", None)
                primed = getattr(info, "currsize", None)
            cache_desc = cache_max if cache_max not in (None, 0) else "unbounded"
            primed_desc = ""
            if primed is not None:
                primed_desc = f", {primed} vector(s) primed"
            print(
                f"Fold {fold_index}/{total_folds}: sentence-transformer '{args.sentence_transformer_model}' embeddings (dimension {sentence_embedding_dim}, "
                f"cache size {cache_desc}{primed_desc})."
            )
        else:
            print(
                f"Fold {fold_index}/{total_folds}: vocabulary size {len(vocab)} (min frequency = {args.min_freq}); max sequence length {max_seq_len} tokens."
            )

        history: list[dict[str, Any]] = []
        pseudo_rounds: list[dict[str, Any]] = []
        self_play_rounds: list[dict[str, Any]] = []
        augmentation_events: list[dict[str, Any]] = []
        pseudo_examples_store: list[dict[str, Any]] = []
        self_play_examples_store: list[tuple[str, str, float]] = []
        global_epoch = 0
        best_state: dict[str, Any] | None = None
        best_val_acc = -float("inf")
        best_entry: dict[str, Any] | None = None
        best_model_source = "model"
        epochs_since_improvement = 0
        optimizer_step_counter = 0
        ema_update_counter = 0
        swa_update_counter = 0
        memory_sweep_counter = 0
        memory_sweep_time_total = 0.0
        memory_sweep_last_free = float("nan")
        memory_sweep_last_reserved = float("nan")
        total_pseudo_added = 0
        total_self_play_added = 0
        total_augmented_examples = 0
        self_play_round_counter = 0

        grad_acc_steps = max(1, args.grad_accumulation_steps)
        augmentation_rng = random.Random(args.seed * 1009 + fold_index)
        self_play_rng = random.Random(args.seed * 4243 + fold_index)
        existing_texts: set[str] = set(train_texts)

        ema_model: nn.Module | None = None
        if args.ema_decay > 0:
            new_ema_model = create_ema_model(
                model,
                args.ema_decay,
                offload_to_cpu=bool(args.ema_offload_cpu and using_cuda),
                offload_dtype=args.ema_offload_dtype_resolved,
            )
            if not (args.ema_offload_cpu and using_cuda):
                new_ema_model.to(device)
            ema_model = new_ema_model
        swa_model: AveragedModel | None = None
        if args.swa_start_epoch > 0:
            new_swa_model = AveragedModel(model)
            new_swa_model.to(device)
            swa_model = new_swa_model
        swa_scheduler_obj: SWALR | None = None

        def run_stage(stage_name: str, epochs: int) -> bool:
            nonlocal global_epoch, best_state, best_val_acc, epochs_since_improvement, best_entry
            nonlocal optimizer_step_counter, ema_update_counter, swa_update_counter, best_model_source
            nonlocal memory_sweep_counter, memory_sweep_time_total, memory_sweep_last_free, memory_sweep_last_reserved
            nonlocal total_augmented_examples, swa_scheduler_obj, train_emotion_vectors, fold_emotion_memory, fold_emotion_config
            nonlocal class_balancer
            nonlocal train_metadata
            if epochs <= 0:
                return False
            augmented_texts, augmented_labels, augmented_weights, augmented_metadata, augmented_count = augment_training_corpus(
                train_texts,
                train_labels,
                train_weights,
                probability=args.augment_probability,
                strategies=augment_strategies,
                max_copies=args.augment_max_copies,
                max_transforms=args.augment_max_transforms,
                rng=augmentation_rng,
                metadata=train_metadata,
            )
            total_augmented_examples += augmented_count
            if augmented_count and args.augment_probability > 0:
                print(
                    f"Fold {fold_index}/{total_folds} – stage '{stage_name}': generated {augmented_count} augmented variants."
                )
            if emotion_enabled and emotion_dim > 0:
                stage_metadata = augmented_metadata if augmented_metadata is not None else train_metadata
                augmented_emotion_vectors = [
                    compose_emotion_features(
                        text,
                        stage_metadata[idx] if idx < len(stage_metadata) else None,
                        lexicon=emotion_lexicon if lexicon_active else None,
                        metadata_encoder=metadata_encoder,
                        lexicon_dim=lexicon_dim,
                        metadata_dim=metadata_dim,
                    )
                    for idx, text in enumerate(augmented_texts)
                ]
            else:
                augmented_emotion_vectors = None

            train_dataset = IntentDataset(
                augmented_texts,
                augmented_labels,
                vocab=vocab,
                vocab_config=vocab_config,
                label_to_idx=label_to_idx,
                max_len=max_seq_len,
                sample_weights=augmented_weights,
                tokenizer=tokenizer_obj,
                tokenizer_cache=tokenizer_cache,
                embedding_model=embedding_fn,
                emotion_vectors=augmented_emotion_vectors,
                emotion_dim=emotion_dim if emotion_enabled else 0,
                pin_memory=dataset_pin_memory,
                target_device=dataset_embedding_target_device,
            )
            if class_balancer is not None and class_balancer.enabled:
                class_balancer.apply(
                    train_labels,
                    train_weights,
                    dataset_examples=train_dataset.examples,
                    idx_to_label=idx_to_label_list,
                )
            train_loader: DataLoader[BatchExample] = create_data_loader(
                train_dataset, batch_size=args.batch_size, shuffle=True
            )

            effective_steps_per_epoch = math.ceil(len(train_loader) / grad_acc_steps) if len(train_loader) else 0
            scheduler, per_batch = create_scheduler(
                optimizer,
                args.scheduler,
                epochs,
                effective_steps_per_epoch,
                effective_lr,
            )

            for _ in range(1, epochs + 1):
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
                    amp_enabled=use_amp,
                    amp_device_type=amp_device_type,
                    max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
                    emotion_config=fold_emotion_config,
                    meta_config=fold_meta_config,
                    neuro_config=fold_neuro_config,
                    discovery_config=fold_discovery_config,
                    transcendent_config=fold_transcendent_config,
                    frontier_config=fold_frontier_config,
                    rdrop_config=fold_rdrop_config,
                    collect_performance_stats=speed_logger.enabled,
                    memory_regulator=vram_regulator,
                    micro_batch_size=micro_batch_size,
                    memory_sweeper=cuda_memory_sweeper,
                )

                optimizer_step_counter += stats["optimizer_steps"]
                ema_update_counter += stats["ema_updates"]
                swa_update_counter += stats["swa_updates"]
                memory_sweep_counter += int(stats.get("memory_sweeps", 0))
                memory_sweep_time_total += float(stats.get("memory_sweep_time", 0.0))
                last_free = stats.get("memory_sweep_last_free")
                if isinstance(last_free, float) and math.isfinite(last_free):
                    memory_sweep_last_free = last_free
                last_reserved = stats.get("memory_sweep_last_reserved")
                if isinstance(last_reserved, float) and math.isfinite(last_reserved):
                    memory_sweep_last_reserved = last_reserved
                if speed_logger.enabled:
                    speed_logger.record_training_epoch(
                        stage=f"fold{fold_index}:{stage_name}",
                        epoch=global_epoch,
                        seconds=float(stats.get("duration", time.perf_counter() - epoch_start)),
                        examples=float(stats.get("examples", 0.0)),
                        tokens=float(stats.get("tokens", 0.0)),
                        batches=float(stats.get("batches", 0.0)),
                        passes=float(stats.get("examples", 0.0)),
                    )

                curriculum_summary: dict[str, Any] | None = None
                if (
                    curriculum_manager is not None
                    and global_epoch >= args.curriculum_start_epoch
                ):
                    curriculum_dataset = IntentDataset(
                        train_texts,
                        train_labels,
                        vocab=vocab,
                        vocab_config=vocab_config,
                        label_to_idx=label_to_idx,
                        max_len=max_seq_len,
                        sample_weights=train_weights,
                        tokenizer=tokenizer_obj,
                        tokenizer_cache=tokenizer_cache,
                        embedding_model=embedding_fn,
                        emotion_vectors=(
                            train_emotion_vectors if (emotion_enabled and emotion_dim > 0) else None
                        ),
                        emotion_dim=emotion_dim if emotion_enabled else 0,
                        pin_memory=dataset_pin_memory,
                        target_device=dataset_embedding_target_device,
                    )
                    curriculum_loader: DataLoader[BatchExample] = create_data_loader(
                        curriculum_dataset,
                        batch_size=args.batch_size,
                    )
                    train_eval = cast(
                        tuple[float, float, list[int], list[int], list[list[float]]],
                        evaluate(
                            model,
                            curriculum_loader,
                            criterion,
                            device,
                            return_details=True,
                            emotion_config=fold_emotion_config,
                            meta_stacker=fold_meta_stacker,
                        ),
                    )
                    _, _, train_targets_detail, _train_predictions_detail, train_probabilities = train_eval
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

                need_class_metrics = class_balancer is not None and class_balancer.enabled
                eval_context: contextlib.AbstractContextManager[None]
                if eval_source == "ema" and hasattr(eval_model, "activate"):
                    eval_context = eval_model.activate(device)  # type: ignore[attr-defined]
                else:
                    eval_context = contextlib.nullcontext()
                with eval_context:
                    if need_class_metrics:
                        val_eval = cast(
                            tuple[float, float, list[int], list[int], list[list[float]]],
                            evaluate(
                                eval_model,
                                val_loader,
                                criterion,
                                device,
                                return_details=True,
                                emotion_config=fold_emotion_config,
                                meta_stacker=fold_meta_stacker,
                            ),
                        )
                        val_loss, val_acc, val_targets_detail, val_predictions_detail, _ = val_eval
                        epoch_class_metrics = compute_classification_metrics(
                            val_targets_detail,
                            val_predictions_detail,
                            label_to_idx=label_to_idx,
                        )
                    else:
                        val_basic = evaluate(
                            eval_model,
                            val_loader,
                            criterion,
                            device,
                            emotion_config=fold_emotion_config,
                            meta_stacker=fold_meta_stacker,
                        )
                        if len(val_basic) == 2:
                            val_loss, val_acc = val_basic
                        else:
                            val_loss = val_basic[0]
                            val_acc = val_basic[1]
                        epoch_class_metrics = None
                current_lr = optimizer.param_groups[0]["lr"]
                history_entry: dict[str, Any] = {
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
                    "rdrop_loss": float(stats.get("rdrop_loss", 0.0)),
                    "rdrop_kl": float(stats.get("rdrop_kl", 0.0)),
                    "rdrop_passes": float(stats.get("rdrop_passes", 0.0)),
                    "rdrop_alpha": float(stats.get("rdrop_alpha", 0.0)),
                    "frontier_loss": float(stats.get("frontier_loss", 0.0)),
                    "frontier_novelty": float(stats.get("frontier_novelty", 0.0)),
                    "frontier_abstraction": float(stats.get("frontier_abstraction", 0.0)),
                    "frontier_transfer": float(stats.get("frontier_transfer", 0.0)),
                    "frontier_curiosity": float(stats.get("frontier_curiosity", 0.0)),
                    "frontier_emotion": float(stats.get("frontier_emotion", 0.0)),
                    "frontier_meta": float(stats.get("frontier_meta", 0.0)),
                    "frontier_diversity": float(stats.get("frontier_diversity", 0.0)),
                    "frontier_updates": float(stats.get("frontier_updates", 0.0)),
                    "moe_loss": float(stats.get("moe_loss", 0.0)),
                    "moe_entropy": float(stats.get("moe_entropy", 0.0)),
                    "moe_entropy_gap": float(stats.get("moe_entropy_gap", 0.0)),
                    "moe_balance": float(stats.get("moe_balance", 0.0)),
                    "moe_active": float(stats.get("moe_active", 0.0)),
                    "moe_max_gate": float(stats.get("moe_max_gate", 0.0)),
                    "moe_batches": float(stats.get("moe_batches", 0.0)),
                    "moe_utilisation_mean": float(stats.get("moe_utilisation_mean", 0.0)),
                    "moe_utilisation_min": float(stats.get("moe_utilisation_min", 0.0)),
                    "moe_utilisation_max": float(stats.get("moe_utilisation_max", 0.0)),
                }
                if epoch_class_metrics is not None:
                    history_entry["val_macro_f1"] = float(epoch_class_metrics.get("macro_f1", 0.0))
                    history_entry["val_macro_precision"] = float(epoch_class_metrics.get("macro_precision", 0.0))
                    history_entry["val_macro_recall"] = float(epoch_class_metrics.get("macro_recall", 0.0))
                balance_stats: dict[str, Any] | None = None
                if class_balancer is not None and class_balancer.enabled and epoch_class_metrics is not None:
                    class_balancer.update(epoch_class_metrics.get("per_label", {}))
                    class_balancer.apply(
                        train_labels,
                        train_weights,
                        dataset_examples=train_dataset.examples,
                        idx_to_label=idx_to_label_list,
                    )
                    balance_stats = class_balancer.stats()
                    history_entry.update(balance_stats)
                history.append(history_entry)

                elapsed = time.perf_counter() - epoch_start
                print(
                    f"Fold {fold_index}/{total_folds} epoch {global_epoch:03d} [{stage_name}] "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"train_acc={train_acc * 100:.2f}% val_acc={val_acc * 100:.2f}% "
                    f"lr={current_lr:.6f} ({elapsed:.1f}s)"
                )
                if balance_stats is not None:
                    print(
                        "   -> class balance multipliers "
                        f"min {balance_stats['class_balance_min']:.2f} "
                        f"max {balance_stats['class_balance_max']:.2f} "
                        f"mean {balance_stats['class_balance_mean']:.2f}"
                    )
                if stats.get("rdrop_loss", 0.0):
                    print(
                        "   -> r-drop kl "
                        f"{stats.get('rdrop_kl', 0.0):.4f} "
                        f"loss {stats.get('rdrop_loss', 0.0):.4f} "
                        f"passes {int(stats.get('rdrop_passes', 0.0))}"
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
                        f"   -> curriculum avg×{curriculum_summary['avg_multiplier']:.2f} "
                        f"(boosted {curriculum_summary['boosted']}, dampened {curriculum_summary['dampened']}, "
                        f"examples {curriculum_summary['examples']}); hardest {preview}"
                    )
                if fold_meta_config is not None and fold_meta_config.enabled:
                    print(
                        "   -> meta-introspection loss "
                        f"{stats.get('meta_loss', 0.0):.4f} "
                        f"gap {stats.get('meta_gap', 0.0):.3f} "
                        f"coverage {stats.get('meta_coverage', 0.0):.2f}"
                    )
                if fold_neuro_config is not None and fold_neuro_config.enabled:
                    print(
                        "   -> neuro-symbolic loss "
                        f"{stats.get('neuro_loss', 0.0):.4f} "
                        f"struct {stats.get('neuro_structural', 0.0):.4f} "
                        f"cohesion {stats.get('neuro_cohesion', 0.0):.3f} "
                        f"entropy {stats.get('neuro_entropy', 0.0):.3f}"
                    )
                if fold_discovery_config is not None and fold_discovery_config.enabled:
                    print(
                        "   -> self-discovery loss "
                        f"{stats.get('discovery_loss', 0.0):.4f} "
                        f"align {stats.get('discovery_alignment', 0.0):.4f} "
                        f"curiosity {stats.get('discovery_curiosity', 0.0):.3f}"
                    )
                if fold_transcendent_config is not None and fold_transcendent_config.enabled:
                    print(
                        "   -> transcendent cognition loss "
                        f"{stats.get('transcendent_loss', 0.0):.4f} "
                        f"coherence {stats.get('transcendent_coherence', 0.0):.3f} "
                        f"stability {stats.get('transcendent_stability', 0.0):.4f}"
                    )
                if fold_frontier_config is not None and fold_frontier_config.enabled:
                    print(
                        "   -> frontier intelligence loss "
                        f"{stats.get('frontier_loss', 0.0):.4f} "
                        f"novelty {stats.get('frontier_novelty', 0.0):.4f} "
                        f"diversity {stats.get('frontier_diversity', 0.0):.3f}"
                    )
                if stats.get("moe_batches", 0.0):
                    print(
                        "   -> mixture-of-experts loss "
                        f"{stats.get('moe_loss', 0.0):.4f} "
                        f"entropy {stats.get('moe_entropy', 0.0):.3f} "
                        f"balance {stats.get('moe_balance', 0.0):.4f} "
                        f"active {stats.get('moe_active', 0.0):.2f} "
                        f"max {stats.get('moe_max_gate', 0.0):.3f}"
                    )
                if stats.get("micro_batch_size"):
                    print(
                        "   -> micro-batching ran {count} slices of ~{size} samples each.".format(
                            count=int(stats.get("micro_batches", 0)),
                            size=int(stats.get("micro_batch_size", 0)),
                        )
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
                accepted_confidences: list[float] = []
                accepted_consistency: list[float] = []
                accepted_margins: list[float] = []
                example_summaries: list[dict[str, Any]] = []
                aggregated_distribution: dict[str, Any] = defaultdict(float)

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
                        if text:
                            text = enrich_with_orion_exploration(text, source_label, self_play_rng)
                        if not text or text in existing_texts:
                            continue
                        attempted += 1
                        pseudo_context: contextlib.AbstractContextManager[None]
                        if pseudo_source is not model and hasattr(pseudo_source, "activate"):
                            pseudo_context = pseudo_source.activate(device)  # type: ignore[attr-defined]
                        else:
                            pseudo_context = contextlib.nullcontext()
                        with pseudo_context:
                            evaluation = evaluate_self_play_candidate(
                                pseudo_source,
                                text,
                                vocab=vocab,
                                label_to_idx=label_to_idx,
                                max_len=max_seq_len,
                                device=device,
                                tokenizer=tokenizer_obj,
                                tokenizer_cache=tokenizer_cache,
                                embedding_model=embedding_fn,
                                samples=args.self_play_samples,
                                vocab_config=vocab_config,
                                emotion_encoder=emotion_lexicon if (emotion_enabled and lexicon_dim > 0) else None,
                                emotion_dim=emotion_dim,
                                emotion_config=fold_emotion_config,
                                metadata_encoder=metadata_encoder if metadata_dim > 0 else None,
                                lexicon_dim=lexicon_dim,
                                metadata_dim=metadata_dim,
                                keyword_calibrator=fold_keyword_calibrator,
                                symbolic_router=fold_cognitive_router,
                                meta_stacker=fold_meta_stacker,
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
                            consistency=float(evaluation.consistency),
                            consistency_floor=float(args.self_play_consistency),
                        )
                        train_texts.append(text)
                        train_labels.append(predicted_label)
                        train_weights.append(weight)
                        train_metadata.append(None)
                        if class_balancer is not None and class_balancer.enabled:
                            class_balancer.register_samples([predicted_label])
                            applied_multiplier = class_balancer.last_applied(predicted_label)
                            train_weights[-1] = float(train_weights[-1] * applied_multiplier)
                        if (
                            emotion_enabled
                            and fold_emotion_memory is not None
                            and emotion_dim > 0
                        ):
                            vector = compose_emotion_features(
                                text,
                                None,
                                lexicon=emotion_lexicon if lexicon_active else None,
                                metadata_encoder=metadata_encoder,
                                lexicon_dim=lexicon_dim,
                                metadata_dim=metadata_dim,
                            )
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
                pseudo_context: contextlib.AbstractContextManager[None]
                if pseudo_source is not model and hasattr(pseudo_source, "activate"):
                    pseudo_context = pseudo_source.activate(device)  # type: ignore[attr-defined]
                else:
                    pseudo_context = contextlib.nullcontext()
                with pseudo_context:
                    confident, unlabeled_texts, pseudo_stats = pseudo_label_unlabeled(
                        pseudo_source,
                        unlabeled_texts,
                        vocab=vocab,
                        label_to_idx=label_to_idx,
                        max_len=max_seq_len,
                        device=device,
                        threshold=current_threshold,
                        vocab_config=vocab_config,
                        tokenizer=tokenizer_obj,
                        tokenizer_cache=tokenizer_cache,
                        embedding_model=embedding_fn,
                        emotion_encoder=emotion_lexicon if (emotion_enabled and emotion_dim > 0) else None,
                        emotion_dim=emotion_dim,
                        emotion_config=fold_emotion_config,
                        consistency_passes=args.self_train_consistency_passes,
                        consistency_max_std=args.self_train_consistency_max_std,
                        consistency_min_agreement=args.self_train_consistency_min_agreement,
                        metadata_encoder=metadata_encoder if metadata_dim > 0 else None,
                        lexicon_dim=lexicon_dim,
                        metadata_dim=metadata_dim,
                        keyword_calibrator=fold_keyword_calibrator,
                        symbolic_router=fold_cognitive_router,
                        meta_stacker=fold_meta_stacker,
                    )
                if not confident:
                    print(
                        f"Fold {fold_index}/{total_folds} self-training round {round_idx}: "
                        f"no predictions met the confidence threshold {current_threshold:.3f}."
                    )
                    if pseudo_stats.get("evaluated"):
                        print(
                            "  -> candidate summary: "
                            f"evaluated {int(pseudo_stats['evaluated'])}"
                            f", rejects (confidence/consistency/agreement) = "
                            f"{int(pseudo_stats.get('reject_confidence', 0))}/"
                            f"{int(pseudo_stats.get('reject_consistency', 0))}/"
                            f"{int(pseudo_stats.get('reject_agreement', 0))}"
                        )
                    continue
                avg_conf = pseudo_stats.get("avg_confidence", 0.0)
                avg_agreement = pseudo_stats.get("avg_agreement", 0.0)
                avg_consistency = pseudo_stats.get("avg_consistency", 0.0)
                avg_std = pseudo_stats.get("avg_std", 0.0)
                pseudo_counts = Counter(decision.label for decision in confident)
                print(
                    f"Fold {fold_index}/{total_folds} self-training round {round_idx}: added {len(confident)} pseudo-labelled examples "
                    f"(avg confidence {avg_conf:.3f}, agreement {avg_agreement:.3f}, "
                    f"consistency {avg_consistency:.3f}, std {avg_std:.4f}, threshold {current_threshold:.3f}). "
                    f"Remaining unlabeled: {len(unlabeled_texts)}"
                )
                added_examples = 0
                for decision in confident:
                    text = decision.text
                    label = decision.label
                    score = decision.confidence
                    weight = compute_pseudo_weight(
                        float(args.self_train_weight),
                        float(score),
                        float(current_threshold),
                        float(args.self_train_confidence_power),
                        float(args.self_train_max_weight_multiplier),
                        consistency=float(decision.consistency),
                        consistency_floor=float(args.self_train_consistency_min_agreement),
                        consistency_power=float(args.self_train_consistency_power),
                    )
                    train_texts.append(text)
                    train_labels.append(label)
                    train_weights.append(weight)
                    train_metadata.append(None)
                    if class_balancer is not None and class_balancer.enabled:
                        class_balancer.register_samples([label])
                        applied_multiplier = class_balancer.last_applied(label)
                        train_weights[-1] = float(train_weights[-1] * applied_multiplier)
                    if (
                        emotion_enabled
                        and fold_emotion_memory is not None
                        and emotion_dim > 0
                    ):
                        vector = compose_emotion_features(
                            text,
                            None,
                            lexicon=emotion_lexicon if lexicon_active else None,
                            metadata_encoder=metadata_encoder,
                            lexicon_dim=lexicon_dim,
                            metadata_dim=metadata_dim,
                        )
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
                    pseudo_examples_store.append(
                        {
                            "text": text,
                            "label": label,
                            "weight": weight,
                            "confidence": score,
                            "consistency": float(decision.consistency),
                            "agreement": float(decision.agreement),
                            "std": float(decision.std),
                        }
                    )
                    added_examples += 1
                total_pseudo_added += added_examples
                pseudo_rounds.append(
                    {
                        "round": float(round_idx),
                        "added_examples": float(added_examples),
                        "average_confidence": float(avg_conf),
                        "average_agreement": float(avg_agreement),
                        "average_consistency": float(avg_consistency),
                        "average_std": float(avg_std),
                        "threshold": float(current_threshold),
                        "label_histogram": dict(sorted(pseudo_counts.items())),
                        "evaluated": float(pseudo_stats.get("evaluated", 0.0)),
                        "reject_confidence": int(pseudo_stats.get("reject_confidence", 0)),
                        "reject_consistency": int(pseudo_stats.get("reject_consistency", 0)),
                        "reject_agreement": int(pseudo_stats.get("reject_agreement", 0)),
                        "consistency_passes": int(pseudo_stats.get("passes", args.self_train_consistency_passes)),
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

        if args.meta_stacker:
            candidate_stacker = MetaIntentStacker(
                label_to_idx=label_to_idx,
                scale=args.meta_stacker_scale,
                regularization=args.meta_stacker_regularization,
                max_iter=args.meta_stacker_max_iter,
                min_accuracy=args.meta_stacker_min_accuracy,
            )
            trained = candidate_stacker.fit_from_dataloader(
                model,
                val_loader,
                device,
                emotion_config=fold_emotion_config,
            )
            fold_meta_metadata = candidate_stacker.export_metadata()
            if trained:
                fold_meta_stacker = candidate_stacker

        val_loss_final, val_acc_final, val_targets, val_predictions, _ = cast(
            tuple[float, float, list[int], list[int], list[list[float]]],
            evaluate(
                model,
                val_loader,
                criterion,
                device,
                return_details=True,
                emotion_config=fold_emotion_config,
                meta_stacker=fold_meta_stacker,
            ),
        )
        val_accuracy_final = float(val_acc_final)
        class_metrics = compute_classification_metrics(val_targets, val_predictions, label_to_idx=label_to_idx)

        if best_entry is None:
            best_entry = history[-1] if history else {
                "epoch": 0.0,
                "stage": "supervised",
                "train_accuracy": 0.0,
                "train_loss": 0.0,
                "val_accuracy": val_accuracy_final,
                "val_loss": val_loss_final,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "evaluation_model": best_model_source,
            }
        assert best_entry is not None
        best_entry.setdefault("final_val_accuracy", val_accuracy_final)

        meta_snapshot: dict[str, Any] | None = None
        if fold_meta_config is not None and fold_meta_config.enabled:
            meta_snapshot = fold_meta_config.introspector.snapshot()

        balance_export: dict[str, Any] | None = None
        balance_stats_final: dict[str, Any] | None = None
        if class_balancer is not None and class_balancer.enabled:
            balance_export = class_balancer.export()
            balance_stats_final = class_balancer.stats()

        moe_entries = [entry for entry in history if entry.get("moe_batches", 0.0)]
        moe_loss_values = [float(entry.get("moe_loss", 0.0)) for entry in moe_entries]
        moe_entropy_values = [float(entry.get("moe_entropy", 0.0)) for entry in moe_entries]
        moe_gap_values = [float(entry.get("moe_entropy_gap", 0.0)) for entry in moe_entries]
        moe_balance_values = [float(entry.get("moe_balance", 0.0)) for entry in moe_entries]
        moe_active_values = [float(entry.get("moe_active", 0.0)) for entry in moe_entries]
        moe_max_values = [float(entry.get("moe_max_gate", 0.0)) for entry in moe_entries]
        moe_util_mean_values = [
            float(entry.get("moe_utilisation_mean", 0.0)) for entry in moe_entries
        ]
        moe_util_min_values = [
            float(entry.get("moe_utilisation_min", 0.0)) for entry in moe_entries
        ]
        moe_util_max_values = [
            float(entry.get("moe_utilisation_max", 0.0)) for entry in moe_entries
        ]

        metadata = {
            "encoder_type": args.encoder_type,
            "model_name": args.model_name,
            "overdrive_profile": bool(args.overdrive_profile),
            "trainer_version": TRAINER_VERSION,
            "memory_guard": _memory_guard_summary(args),
            "cuda_memory_guard": _cuda_memory_guard_summary(args),
            "performance_overdrive": _performance_overdrive_summary(args),
            "dataset_path": str(args.dataset),
            "dataset_checksum": dataset_checksum,
            "dataset_examples": len(texts),
            "num_labels": num_classes,
            "vocab": vocab if args.encoder_type == "bilstm" else None,
            "vocab_settings": {
                "include_bigrams": bool(vocab_config.include_bigrams),
                "include_trigrams": bool(vocab_config.include_trigrams),
                "include_char_ngrams": bool(vocab_config.include_char_ngrams),
                "char_ngram_min": int(vocab_config.char_ngram_min),
                "char_ngram_max": int(vocab_config.char_ngram_max),
                "char_ngram_limit": int(vocab_config.char_ngram_limit),
                "max_size": int(args.vocab_max_size) if args.vocab_max_size > 0 else None,
                "extra_sources": sorted(active_vocab_sources),
                "extra_fragments": len(deduped_extra_texts) if deduped_extra_texts is not None else 0,
            },
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
            "bilstm_conv_head": bool(args.bilstm_conv_head if args.encoder_type == "bilstm" else False),
            "bilstm_conv_kernels": bilstm_conv_kernel_sizes if args.encoder_type == "bilstm" else [],
            "bilstm_conv_channels": (
                int(args.bilstm_conv_channels) if args.encoder_type == "bilstm" else 0
            ),
            "bilstm_conv_dropout": (
                float(args.bilstm_conv_dropout) if args.encoder_type == "bilstm" else 0.0
            ),
            "test_ratio": args.test_ratio if total_folds == 1 else None,
            "seed": args.seed,
            "compute_device": {
                "type": device.type,
                "index": device.index,
                "name": device_info.get("name"),
                "requested": device_info.get("requested"),
                "descriptor": device_info.get("device"),
                "fallback": device_info.get("fallback"),
                "using_cuda": using_cuda,
                "using_mps": using_mps,
                "available_cuda": available_backends.get("cuda", []),
                "available_mps": available_backends.get("mps", []),
                "memory_gb": device_info.get("memory_gb"),
                "memory_bytes": device_info.get("memory_bytes"),
            },
            "dataloader": {
                "cpu_workers": dataloader_workers,
                "prefetch_factor": dataloader_prefetch,
                "pin_memory": using_cuda,
            },
            "auto_optimizations": list(auto_actions),
            "scheduler": args.scheduler,
            "learning_rate": effective_lr,
            "transformer_model": args.transformer_model if args.encoder_type == "transformer" else None,
            "transformer_layerwise_decay": (
                args.transformer_layerwise_decay if args.encoder_type == "transformer" else None
            ),
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "label_smoothing": args.label_smoothing,
            "use_fp16": bool(use_amp),
            "metadata_feature_strategy": args.metadata_feature_strategy,
            "metadata_feature_dim": metadata_dim,
            "metadata_fields": metadata_encoder.export() if metadata_encoder is not None else None,
            "lexicon_feature_dim": lexicon_dim,
            "emotion_feature_dim": emotion_dim,
            "training_examples_supervised": base_train_size,
            "training_examples_final": len(train_texts),
            "validation_examples": len(val_texts),
            "promotion_tolerance": args.promotion_tolerance,
            "class_distribution_total": dict(sorted(Counter(labels).items())),
            "rdrop_alpha": float(args.rdrop_alpha),
            "rdrop_forward_passes": int(args.rdrop_forward_passes),
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
                "consistency_passes": args.self_train_consistency_passes,
                "consistency_max_std": args.self_train_consistency_max_std,
                "consistency_min_agreement": args.self_train_consistency_min_agreement,
                "consistency_power": args.self_train_consistency_power,
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
                "ema_offload_cpu": bool(args.ema_offload_cpu and using_cuda),
                "ema_offload_dtype": args.ema_offload_dtype,
                "swa_start_epoch": args.swa_start_epoch,
                "swa_lr": args.swa_lr if args.swa_lr > 0 else effective_lr,
                "swa_anneal_epochs": args.swa_anneal_epochs,
                "swa_updates": swa_update_counter,
                "optimizer_steps": optimizer_step_counter,
                "memory_sweeps": memory_sweep_counter,
                "memory_sweep_time": memory_sweep_time_total,
                "memory_sweep_last_free": memory_sweep_last_free,
                "memory_sweep_last_reserved": memory_sweep_last_reserved,
                "best_model_source": best_model_source,
                "rdrop_alpha": float(args.rdrop_alpha),
                "rdrop_forward_passes": int(args.rdrop_forward_passes),
                "transformer_layerwise_decay": (
                    float(args.transformer_layerwise_decay)
                    if args.encoder_type == "transformer"
                    else None
                ),
                "micro_batch_size": int(args.micro_batch_size_effective),
                "micro_batch_regulator": vram_regulator.export() if vram_regulator is not None else None,
                "cuda_memory_sweeper": (
                    cuda_memory_sweeper.export() if cuda_memory_sweeper is not None else None
                ),
            },
            "adaptive_curriculum": (
                curriculum_manager.export_metadata()
                if curriculum_manager is not None
                else {"enabled": False}
            ),
            "class_balance": (
                {
                    "enabled": True,
                    "strategy": args.class_balance_strategy,
                    "boost": args.class_balance_boost,
                    "power": args.class_balance_power,
                    "momentum": args.class_balance_momentum,
                    "min_multiplier": args.class_balance_min_multiplier,
                    "max_multiplier": args.class_balance_max_multiplier,
                    "floor": args.class_balance_floor,
                    "min_support": args.class_balance_min_support,
                    "state": balance_export,
                }
                if balance_export is not None
                else {
                    "enabled": args.class_balance_strategy != "none",
                    "strategy": args.class_balance_strategy,
                }
            ),
            "keyword_calibration": fold_keyword_metadata,
            "meta_stacker": fold_meta_metadata,
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
            "metadata_feature_strategy": args.metadata_feature_strategy,
            "metadata_feature_dim": metadata_dim,
            "lexicon_feature_dim": lexicon_dim,
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
            metadata["sentence_transformer_mlp_layers"] = args.st_mlp_layers
            metadata["sentence_transformer_mlp_expansion"] = args.st_mlp_expansion
            metadata["sentence_transformer_hidden_dims"] = list(args.st_mlp_hidden_dims)
            metadata["sentence_transformer_activation"] = args.st_mlp_activation
            metadata["sentence_transformer_final_dropout"] = args.st_final_dropout
            metadata["sentence_transformer_layer_norm"] = bool(args.st_mlp_layer_norm)
            metadata["sentence_transformer_residual"] = bool(args.st_mlp_residual)
            metadata["sentence_transformer_moe_enabled"] = bool(args.st_moe_enabled)
            metadata["sentence_transformer_moe_experts"] = int(args.st_moe_experts)
            metadata["sentence_transformer_moe_hidden_dim"] = int(
                args.st_moe_effective_hidden_dim if args.st_moe_enabled else 0
            )
            metadata["sentence_transformer_moe_activation"] = args.st_moe_activation
            metadata["sentence_transformer_moe_dropout"] = args.st_moe_dropout
            metadata["sentence_transformer_moe_temperature"] = args.st_moe_temperature
            metadata["sentence_transformer_moe_topk"] = int(args.st_moe_topk)
            metadata["sentence_transformer_moe_entropy_weight"] = args.st_moe_entropy_weight
            metadata["sentence_transformer_moe_balance_weight"] = args.st_moe_balance_weight
            metadata["sentence_transformer_moe_layer_norm"] = bool(args.st_moe_layer_norm)
            metadata["sentence_transformer_moe_utilisation_momentum"] = args.st_moe_utilisation_momentum

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
        moe_entries = [entry for entry in history if entry.get("moe_batches", 0.0)]
        moe_loss_values = [float(entry.get("moe_loss", 0.0)) for entry in moe_entries]
        moe_entropy_values = [float(entry.get("moe_entropy", 0.0)) for entry in moe_entries]
        moe_gap_values = [float(entry.get("moe_entropy_gap", 0.0)) for entry in moe_entries]
        moe_balance_values = [float(entry.get("moe_balance", 0.0)) for entry in moe_entries]
        moe_active_values = [float(entry.get("moe_active", 0.0)) for entry in moe_entries]
        moe_max_values = [float(entry.get("moe_max_gate", 0.0)) for entry in moe_entries]
        moe_util_mean_values = [
            float(entry.get("moe_utilisation_mean", 0.0)) for entry in moe_entries
        ]
        moe_util_min_values = [
            float(entry.get("moe_utilisation_min", 0.0)) for entry in moe_entries
        ]
        moe_util_max_values = [
            float(entry.get("moe_utilisation_max", 0.0)) for entry in moe_entries
        ]

        metrics: dict[str, Any] = {
            "model_name": args.model_name,
            "trainer_version": TRAINER_VERSION,
            "memory_guard": _memory_guard_summary(args),
            "cuda_memory_guard": _cuda_memory_guard_summary(args),
            "performance_overdrive": _performance_overdrive_summary(args),
            "timestamp_utc": datetime.now(UTC_ZONE).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "overdrive_profile": bool(args.overdrive_profile),
            "validation_accuracy": float(best_val_acc),
            "train_accuracy_at_best": float(best_entry["train_accuracy"]),
            "best_epoch": float(best_entry["epoch"]),
            "best_stage": best_entry["stage"],
            "epochs_ran": float(global_epoch),
            "dataset_examples": len(texts),
            "dataset_checksum": dataset_checksum,
            "num_labels": num_classes,
            "vocab_size": len(vocab),
            "vocab_include_bigrams": bool(vocab_config.include_bigrams),
            "vocab_include_trigrams": bool(vocab_config.include_trigrams),
            "vocab_include_char_ngrams": bool(vocab_config.include_char_ngrams),
            "vocab_char_ngram_min": int(vocab_config.char_ngram_min),
            "vocab_char_ngram_max": int(vocab_config.char_ngram_max),
            "vocab_char_ngram_limit": int(vocab_config.char_ngram_limit),
            "vocab_max_size": int(args.vocab_max_size) if args.vocab_max_size > 0 else None,
            "vocab_extra_fragments": len(deduped_extra_texts) if deduped_extra_texts is not None else 0,
            "vocab_extra_sources": sorted(active_vocab_sources),
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
            "keyword_calibration_enabled": bool(fold_keyword_calibrator is not None),
            "keyword_calibration_feature_count": int(
                fold_keyword_calibrator.feature_count if fold_keyword_calibrator is not None else 0
            ),
            "cognitive_router_enabled": bool(fold_cognitive_router is not None),
            "cognitive_router_trigger_total": int(
                fold_cognitive_router.total_triggers if fold_cognitive_router is not None else 0
            ),
            "cognitive_router_examples": int(
                fold_cognitive_router.adjusted_examples if fold_cognitive_router is not None else 0
            ),
            "meta_stacker_enabled": bool(fold_meta_stacker is not None),
            "meta_stacker_training_accuracy": float(
                fold_meta_metadata.get("training_accuracy", 0.0)
            ),
            "meta_stacker_trained_samples": int(fold_meta_metadata.get("trained_samples", 0)),
            "meta_stacker_feature_count": int(fold_meta_metadata.get("feature_count", 0)),
        }
        monitor_snapshot = hardware_monitor.snapshot()
        metrics["hardware_monitor"] = monitor_snapshot
        if overdrive_simulation_summary:
            metrics["overdrive_simulation"] = overdrive_simulation_summary
        if moe_loss_values:
            metrics["moe_loss_mean"] = float(sum(moe_loss_values) / len(moe_loss_values))
            metrics["moe_loss_last"] = float(moe_loss_values[-1])
        if moe_entropy_values:
            metrics["moe_entropy_mean"] = float(sum(moe_entropy_values) / len(moe_entropy_values))
            metrics["moe_entropy_last"] = float(moe_entropy_values[-1])
        if moe_gap_values:
            metrics["moe_entropy_gap_mean"] = float(sum(moe_gap_values) / len(moe_gap_values))
            metrics["moe_entropy_gap_last"] = float(moe_gap_values[-1])
        if moe_balance_values:
            metrics["moe_balance_mean"] = float(sum(moe_balance_values) / len(moe_balance_values))
            metrics["moe_balance_last"] = float(moe_balance_values[-1])
        if moe_active_values:
            metrics["moe_active_mean"] = float(sum(moe_active_values) / len(moe_active_values))
            metrics["moe_active_last"] = float(moe_active_values[-1])
        if moe_max_values:
            metrics["moe_max_gate_mean"] = float(sum(moe_max_values) / len(moe_max_values))
            metrics["moe_max_gate_last"] = float(moe_max_values[-1])
        if moe_util_mean_values:
            metrics["moe_utilisation_mean_mean"] = float(
                sum(moe_util_mean_values) / len(moe_util_mean_values)
            )
            metrics["moe_utilisation_mean_last"] = float(moe_util_mean_values[-1])
        if moe_util_min_values:
            metrics["moe_utilisation_min_mean"] = float(
                sum(moe_util_min_values) / len(moe_util_min_values)
            )
            metrics["moe_utilisation_min_last"] = float(moe_util_min_values[-1])
        if moe_util_max_values:
            metrics["moe_utilisation_max_mean"] = float(
                sum(moe_util_max_values) / len(moe_util_max_values)
            )
            metrics["moe_utilisation_max_last"] = float(moe_util_max_values[-1])
        if moe_entries:
            metrics["moe_batches_last"] = float(moe_entries[-1].get("moe_batches", 0.0))

        metrics["class_balance_strategy"] = args.class_balance_strategy
        metrics["class_balance_enabled"] = bool(args.class_balance_strategy != "none")
        if balance_stats_final is not None:
            metrics["class_balance_min_multiplier"] = float(balance_stats_final["class_balance_min"])
            metrics["class_balance_max_multiplier"] = float(balance_stats_final["class_balance_max"])
            metrics["class_balance_mean_multiplier"] = float(balance_stats_final["class_balance_mean"])
        elif args.class_balance_strategy != "none":
            metrics["class_balance_min_multiplier"] = None
            metrics["class_balance_max_multiplier"] = None
            metrics["class_balance_mean_multiplier"] = None
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
            metrics["transformer_layerwise_decay"] = float(args.transformer_layerwise_decay)
            metrics["rdrop_alpha"] = float(args.rdrop_alpha)
            metrics["rdrop_forward_passes"] = int(args.rdrop_forward_passes)
            if tokenizer_obj is not None:
                metrics["tokenizer_vocab_size"] = len(tokenizer_obj)
        if args.encoder_type == "st":
            metrics["sentence_transformer_model"] = args.sentence_transformer_model
            metrics["sentence_transformer_hidden_dim"] = args.st_hidden_dim
            metrics["sentence_transformer_dropout"] = args.st_dropout
            metrics["sentence_transformer_mlp_layers"] = args.st_mlp_layers
            metrics["sentence_transformer_mlp_expansion"] = args.st_mlp_expansion
            metrics["sentence_transformer_hidden_dims"] = list(args.st_mlp_hidden_dims)
            metrics["sentence_transformer_activation"] = args.st_mlp_activation
            metrics["sentence_transformer_final_dropout"] = args.st_final_dropout
            metrics["sentence_transformer_layer_norm"] = bool(args.st_mlp_layer_norm)
            metrics["sentence_transformer_residual"] = bool(args.st_mlp_residual)
            metrics["sentence_transformer_moe_enabled"] = bool(args.st_moe_enabled)
            metrics["sentence_transformer_moe_experts"] = int(args.st_moe_experts)
            metrics["sentence_transformer_moe_hidden_dim"] = int(
                args.st_moe_effective_hidden_dim if args.st_moe_enabled else 0
            )
            metrics["sentence_transformer_moe_activation"] = args.st_moe_activation
            metrics["sentence_transformer_moe_dropout"] = args.st_moe_dropout
            metrics["sentence_transformer_moe_temperature"] = args.st_moe_temperature
            metrics["sentence_transformer_moe_topk"] = int(args.st_moe_topk)
            metrics["sentence_transformer_moe_entropy_weight"] = args.st_moe_entropy_weight
            metrics["sentence_transformer_moe_balance_weight"] = args.st_moe_balance_weight
            metrics["sentence_transformer_moe_layer_norm"] = bool(args.st_moe_layer_norm)
            metrics["sentence_transformer_moe_utilisation_momentum"] = args.st_moe_utilisation_momentum
            metrics["sentence_transformer_moe_enabled"] = bool(args.st_moe_enabled)
            metrics["sentence_transformer_moe_experts"] = int(args.st_moe_experts)
            metrics["sentence_transformer_moe_hidden_dim"] = int(
                args.st_moe_effective_hidden_dim if args.st_moe_enabled else 0
            )
            metrics["sentence_transformer_moe_activation"] = args.st_moe_activation
            metrics["sentence_transformer_moe_dropout"] = args.st_moe_dropout
            metrics["sentence_transformer_moe_temperature"] = args.st_moe_temperature
            metrics["sentence_transformer_moe_topk"] = int(args.st_moe_topk)
            metrics["sentence_transformer_moe_entropy_weight"] = args.st_moe_entropy_weight
            metrics["sentence_transformer_moe_balance_weight"] = args.st_moe_balance_weight
            metrics["sentence_transformer_moe_layer_norm"] = bool(args.st_moe_layer_norm)
            metrics["sentence_transformer_moe_utilisation_momentum"] = args.st_moe_utilisation_momentum
        if curriculum_manager is not None:
            metrics.update(curriculum_manager.export_metrics())

        evaluation_outputs: list[dict[str, Any]] = []
        for sample in evaluation_inputs:
            analysis_features = inspect_text_characteristics(sample)
            prediction = predict_with_trace(
                model,
                sample,
                vocab=vocab,
                label_to_idx=label_to_idx,
                max_len=max_seq_len,
                device=device,
                tokenizer=tokenizer_obj,
                tokenizer_cache=tokenizer_cache,
                embedding_model=embedding_fn,
                vocab_config=vocab_config,
                emotion_encoder=emotion_lexicon if (emotion_enabled and emotion_dim > 0) else None,
                emotion_dim=emotion_dim,
                emotion_config=fold_emotion_config,
                metadata=None,
                metadata_encoder=metadata_encoder if metadata_dim > 0 else None,
                lexicon_dim=lexicon_dim,
                metadata_dim=metadata_dim,
                calibrator=fold_keyword_calibrator,
                symbolic_router=fold_cognitive_router,
                meta_stacker=fold_meta_stacker,
            )
            response = generate_response(prediction.label, sample)
            valuation_summary: dict[str, Any] = {
                "question_type": analysis_features.get("question_type"),
                "false_question": bool(analysis_features.get("false_question")),
                "seduction_style": analysis_features.get("seduction_style"),
                "comma_count": int(analysis_features.get("comma_count", 0)),
                "uppercase_ratio": round(float(analysis_features.get("uppercase_ratio", 0.0)), 3),
            }
            uppercase_tokens = list(analysis_features.get("uppercase_tokens", []))
            if uppercase_tokens:
                valuation_summary["uppercase_tokens"] = uppercase_tokens[:3]
            seduction_terms = list(analysis_features.get("seduction_terms", []))
            if seduction_terms:
                valuation_summary["seduction_terms"] = sorted(set(seduction_terms))[:3]
            valuation_rng = _orion_seed_rng(f"valuation::{sample}")
            valuation_reflections = craft_orion_reflections(
                analysis_features,
                label=prediction.label,
                rng=valuation_rng,
                context="valuation",
            )[:3]
            entry: dict[str, Any] = {
                "input": sample,
                "predicted_intent": prediction.label,
                "confidence": prediction.confidence,
                "top_intents": [
                    {"label": label, "confidence": score}
                    for label, score in prediction.top_predictions
                ],
                "response": response.message,
                "response_strategy": response.strategy,
                "valuation": valuation_summary,
            }
            if response.basis:
                entry["response_basis"] = response.basis
            if valuation_reflections:
                entry["valuation_reflections"] = valuation_reflections
            evaluation_outputs.append(entry)

        if (
            fold_cognitive_router is not None
            and fold_cognitive_router.total_triggers > 0
        ):
            print(
                f"Fold {fold_index}/{total_folds}: cognitive router emitted "
                f"{fold_cognitive_router.total_triggers} adjustments across "
                f"{fold_cognitive_router.adjusted_examples} analysed texts."
            )

        model.cpu()
        if using_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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
    def run_full_dataset_training(best_fold: FoldResult) -> dict[str, Any] | None:
        nonlocal fp16_warning_emitted
        if args.final_train_epochs <= 0:
            return None

        final_texts = list(texts)
        final_labels = list(labels)
        final_weights = [1.0] * len(final_texts)
        final_metadata: list[dict[str, str] | None] = [
            metadata_rows[i] if i < len(metadata_rows) else None for i in range(len(final_texts))
        ]
        final_keyword_calibrator: KeywordIntentCalibrator | None = None
        if args.keyword_calibration:
            final_keyword_calibrator = build_keyword_intent_calibrator(
                final_texts,
                final_labels,
                label_to_idx=label_to_idx,
                min_frequency=args.keyword_calibration_min_frequency,
                bigram_min_frequency=args.keyword_calibration_bigram_min_frequency,
                smoothing=args.keyword_calibration_smoothing,
                strength_threshold=args.keyword_calibration_strength_threshold,
                max_features_per_label=args.keyword_calibration_max_features,
                bias_weight=args.keyword_calibration_bias_weight,
                feature_weight=args.keyword_calibration_feature_weight,
                normalise_power=args.keyword_calibration_normalise_power,
            )
        final_cognitive_router: CognitiveIntentRouter | None = None
        if args.cognitive_router:
            final_cognitive_router = CognitiveIntentRouter(
                label_to_idx=label_to_idx,
                signal_scale=args.cognitive_router_signal_scale,
                penalty_scale=args.cognitive_router_penalty_scale,
                synergy_scale=args.cognitive_router_synergy_scale,
            )
        final_meta_stacker: MetaIntentStacker | None = None
        final_meta_metadata: dict[str, Any] = {"enabled": bool(args.meta_stacker)}
        base_final_keyword_metadata = (
            final_keyword_calibrator.export_metadata()
            if final_keyword_calibrator is not None
            else {"enabled": bool(args.keyword_calibration)}
        )
        final_keyword_metadata = _snapshot_mapping(base_final_keyword_metadata) or {
            "enabled": bool(args.keyword_calibration)
        }
        if final_cognitive_router is not None:
            router_export = final_cognitive_router.export_metadata()
            final_router_metadata = _snapshot_mapping(router_export) or {"enabled": True}
        else:
            final_router_metadata = {"enabled": False}
        final_keyword_metadata["router"] = final_router_metadata
        final_keyword_metadata["router_enabled"] = bool(final_router_metadata.get("enabled", False))
        final_keyword_metadata["enabled"] = bool(
            final_keyword_metadata.get("enabled", False) or final_router_metadata.get("enabled", False)
        )
        final_keyword_metadata["router_trigger_total"] = int(
            final_router_metadata.get("positive_trigger_events", 0)
            + final_router_metadata.get("negative_trigger_events", 0)
        )
        final_emotion_vectors: list[list[float]] = []
        final_emotion_memory: EmotionPrototypeMemory | None = None
        final_emotion_config: EmotionTrainingConfig | None = None
        if emotion_enabled and emotion_dim > 0:
            final_emotion_vectors = [
                compose_emotion_features(
                    text,
                    final_metadata[idx] if idx < len(final_metadata) else None,
                    lexicon=emotion_lexicon if lexicon_active else None,
                    metadata_encoder=metadata_encoder,
                    lexicon_dim=lexicon_dim,
                    metadata_dim=metadata_dim,
                )
                for idx, text in enumerate(final_texts)
            ]
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
        final_rdrop_config: RDropConfig | None = None
        if args.rdrop_alpha > 0:
            final_rdrop_config = RDropConfig(
                enabled=True,
                alpha=float(args.rdrop_alpha),
                passes=max(2, int(args.rdrop_forward_passes)),
            )
        final_meta_config: MetaCognitiveConfig | None = None
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
            meta_section = _metadata_section(best_fold.metadata, "meta_introspection")
            if meta_section is not None:
                snapshot_obj = meta_section.get("snapshot")
                meta_snapshot = _snapshot_mapping(snapshot_obj)
                if meta_snapshot is not None:
                    try:
                        final_meta_config.introspector.load_snapshot(meta_snapshot, device)
                    except Exception:
                        pass
        final_neuro_config: NeuroSymbolicConfig | None = None
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
            neuro_section = _metadata_section(best_fold.metadata, "neuro_symbolic")
            best_neuro_snapshot = None
            if neuro_section is not None:
                snapshot_obj = neuro_section.get("snapshot")
                best_neuro_snapshot = _snapshot_mapping(snapshot_obj)
            if best_neuro_snapshot is not None:
                try:
                    final_neuro_config.reasoner.load_snapshot(best_neuro_snapshot)
                except Exception:
                    pass
        final_discovery_config: SelfDiscoveryConfig | None = None
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
            discovery_section = _metadata_section(best_fold.metadata, "self_discovery")
            best_discovery_snapshot = None
            if discovery_section is not None:
                snapshot_obj = discovery_section.get("snapshot")
                best_discovery_snapshot = _snapshot_mapping(snapshot_obj)
            if best_discovery_snapshot is not None:
                try:
                    final_discovery_config.orchestrator.load_snapshot(best_discovery_snapshot)
                except Exception:
                    pass
        final_transcendent_config: TranscendentCognitionConfig | None = None
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
            transcendent_section = _metadata_section(best_fold.metadata, "transcendent_cognition")
            best_transcendent_snapshot = None
            if transcendent_section is not None:
                snapshot_obj = transcendent_section.get("snapshot")
                best_transcendent_snapshot = _snapshot_mapping(snapshot_obj)
            if best_transcendent_snapshot is not None:
                try:
                    final_transcendent_config.architect.load_snapshot(best_transcendent_snapshot)
                except Exception:
                    pass
        final_frontier_config: FrontierIntelligenceConfig | None = None
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
            frontier_section = _metadata_section(best_fold.metadata, "frontier_intelligence")
            best_frontier_snapshot = None
            if frontier_section is not None:
                snapshot_obj = frontier_section.get("snapshot")
                best_frontier_snapshot = _snapshot_mapping(snapshot_obj)
            if best_frontier_snapshot is not None:
                try:
                    final_frontier_config.catalyst.load_snapshot(best_frontier_snapshot)
                except Exception:
                    pass
        pseudo_total = 0
        synthetic_total = 0
        if args.final_use_pseudo:
            pseudo_cache: dict[str, tuple[str, float]] = {}
            pseudo_seen: set[str] = set()
            synthetic_seen: set[str] = set()
            for result in fold_results:
                for entry in result.pseudo_examples:
                    text = str(entry.get("text", ""))
                    label = entry.get("label")
                    weight = float(entry.get("weight", args.self_train_weight))
                    if not text or not isinstance(label, str):
                        continue
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
                final_metadata.append(None)
                if (
                    emotion_enabled
                    and final_emotion_memory is not None
                    and emotion_dim > 0
                ):
                    vector = compose_emotion_features(
                        text,
                        None,
                        lexicon=emotion_lexicon if lexicon_active else None,
                        metadata_encoder=metadata_encoder,
                        lexicon_dim=lexicon_dim,
                        metadata_dim=metadata_dim,
                    )
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

        amp_backend_available = bool(GradScaler is not None)
        if using_mps:
            amp_backend_available = amp_backend_available or _mps_backend_available()
        use_amp = bool(args.fp16 and (using_cuda or using_mps) and amp_backend_available)
        if args.fp16 and not (using_cuda or using_mps) and not fp16_warning_emitted:
            print(
                f"fp16 requested but the active device '{device.type}' does not support GPU/MPS AMP; "
                "training with full precision for the final stage."
            )
            fp16_warning_emitted = True
        elif args.fp16 and (using_cuda or using_mps) and not amp_backend_available and not fp16_warning_emitted:
            print("fp16 requested but AMP utilities are unavailable; training with full precision for final stage.")
            fp16_warning_emitted = True
        final_scaler = create_grad_scaler(use_amp and amp_device_type == "cuda", amp_device_type)

        criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=args.label_smoothing)
        if args.encoder_type == "transformer":
            optimizer = create_transformer_optimizer(
                final_model,
                base_lr=final_lr,
                weight_decay=final_weight_decay,
                layerwise_decay=float(args.transformer_layerwise_decay),
            )
        else:
            optimizer = _memory_efficient_optimizer(
                final_model.parameters(),
                final_lr,
                final_weight_decay,
            )
        if args.encoder_type == "transformer":
            lr_values = sorted({float(group.get("lr", final_lr)) for group in optimizer.param_groups})
            if len(lr_values) > 1:
                print(
                    "Final stage: layer-wise learning rate span "
                    f"{lr_values[0]:.2e} – {lr_values[-1]:.2e} "
                    f"(decay {args.transformer_layerwise_decay:.3f})"
                )

        grad_acc_steps = max(1, args.grad_accumulation_steps)
        augmentation_rng = random.Random(args.seed * 173 + 2048)
        ema_model: nn.Module | None = None
        if args.ema_decay > 0:
            new_ema_model = create_ema_model(
                final_model,
                args.ema_decay,
                offload_to_cpu=bool(args.ema_offload_cpu and using_cuda),
                offload_dtype=args.ema_offload_dtype_resolved,
            )
            if not (args.ema_offload_cpu and using_cuda):
                new_ema_model.to(device)
            ema_model = new_ema_model
        swa_model: AveragedModel | None = None
        if args.swa_start_epoch > 0:
            new_swa_model = AveragedModel(final_model)
            new_swa_model.to(device)
            swa_model = new_swa_model
        swa_scheduler: SWALR | None = None

        needs_teacher = args.distill_epochs > 0 or args.distill_keep_during_final
        distillation_stats: dict[str, Any] = {
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
        distillation_logits: list[list[float] | None] | None = None
        distillation_weights: list[float] | None = None
        distillation_config: DistillationConfig | None = None
        teacher_logits_for_final: list[list[float] | None] | None = None
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
                    vocab_config=vocab_config,
                    label_to_idx=label_to_idx,
                    max_len=max_seq_len,
                    sample_weights=final_weights,
                    tokenizer=tokenizer_obj,
                    tokenizer_cache=tokenizer_cache,
                    embedding_model=embedding_fn,
                    emotion_vectors=final_emotion_vectors if (emotion_enabled and emotion_dim > 0) else None,
                    emotion_dim=emotion_dim if emotion_enabled else 0,
                    pin_memory=dataset_pin_memory,
                    target_device=dataset_embedding_target_device,
                )
                teacher_loader: DataLoader[BatchExample] = create_data_loader(
                    teacher_dataset,
                    batch_size=final_batch_size,
                    shuffle=False,
                )
                aggregated_logits: list[list[float]] | None = None
                non_blocking_teacher = device.type == "cuda"
                for teacher_result in teacher_pool:
                    teacher_model = build_model()
                    teacher_model.load_state_dict(teacher_result.model_state)
                    teacher_model.to(device)
                    teacher_model.eval()
                    collected_logits: list[list[float]] = []
                    with torch.no_grad():
                        for batch in teacher_loader:
                            if len(batch) == 5:
                                inputs_obj, _labels, _weights, attention_obj, _teacher_logits = batch
                            else:
                                inputs_obj, _labels, _weights, attention_obj = batch  # type: ignore[misc]
                            inputs = _ensure_tensor(
                                inputs_obj, device=device, non_blocking=non_blocking_teacher
                            )
                            attention_mask = _ensure_tensor(
                                attention_obj, device=device, non_blocking=non_blocking_teacher
                            )
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
                    if using_cuda and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if aggregated_logits is not None:
                    teacher_count = max(1, len(teacher_pool))
                    averaged_tensor = torch.tensor(aggregated_logits, dtype=torch.float32) / float(teacher_count)
                    probability_tensor = torch.softmax(averaged_tensor, dim=1)
                    confidence_values, _ = probability_tensor.max(dim=1)
                    averaged_logits = averaged_tensor.tolist()
                    distillation_logits = []
                    distillation_weights = []
                    active_confidences: list[float] = []
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

        augmented_texts, augmented_labels, augmented_weights, augmented_metadata, augmented_count = augment_training_corpus(
            final_texts,
            final_labels,
            final_weights,
            probability=args.augment_probability,
            strategies=augment_strategies,
            max_copies=args.augment_max_copies,
            max_transforms=args.augment_max_transforms,
            rng=augmentation_rng,
            metadata=final_metadata,
        )
        teacher_logits_for_final_dataset: list[list[float] | None] | None = None
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

        if emotion_enabled and emotion_dim > 0:
            stage_metadata = augmented_metadata if augmented_metadata is not None else final_metadata
            augmented_emotion_vectors = [
                compose_emotion_features(
                    text,
                    stage_metadata[idx] if idx < len(stage_metadata) else None,
                    lexicon=emotion_lexicon if lexicon_active else None,
                    metadata_encoder=metadata_encoder,
                    lexicon_dim=lexicon_dim,
                    metadata_dim=metadata_dim,
                )
                for idx, text in enumerate(augmented_texts)
            ]
        else:
            augmented_emotion_vectors = None

        train_dataset = IntentDataset(
            augmented_texts,
            augmented_labels,
            vocab=vocab,
            vocab_config=vocab_config,
            label_to_idx=label_to_idx,
            max_len=max_seq_len,
            sample_weights=augmented_weights,
            tokenizer=tokenizer_obj,
            tokenizer_cache=tokenizer_cache,
            embedding_model=embedding_fn,
            teacher_logits=teacher_logits_for_final_dataset,
            emotion_vectors=augmented_emotion_vectors,
            emotion_dim=emotion_dim if emotion_enabled else 0,
            pin_memory=dataset_pin_memory,
            target_device=dataset_embedding_target_device,
        )
        train_loader: DataLoader[BatchExample] = create_data_loader(
            train_dataset,
            batch_size=final_batch_size,
            shuffle=True,
        )
        eval_keyword_vectors: list[list[float]] | None = None
        if final_keyword_calibrator is not None or final_cognitive_router is not None:
            eval_keyword_vectors = [
                compose_logit_adjustments(
                    text,
                    calibrator=final_keyword_calibrator,
                    router=final_cognitive_router,
                )
                for text in augmented_texts
            ]
        eval_dataset = IntentDataset(
            augmented_texts,
            augmented_labels,
            vocab=vocab,
            vocab_config=vocab_config,
            label_to_idx=label_to_idx,
            max_len=max_seq_len,
            sample_weights=augmented_weights,
            tokenizer=tokenizer_obj,
            tokenizer_cache=tokenizer_cache,
            embedding_model=embedding_fn,
            teacher_logits=teacher_logits_for_final_dataset,
            emotion_vectors=augmented_emotion_vectors,
            emotion_dim=emotion_dim if emotion_enabled else 0,
            keyword_vectors=eval_keyword_vectors,
            pin_memory=dataset_pin_memory,
            target_device=dataset_embedding_target_device,
        )
        eval_loader: DataLoader[BatchExample] = create_data_loader(
            eval_dataset,
            batch_size=final_batch_size,
        )
        effective_steps_per_epoch = math.ceil(len(train_loader) / grad_acc_steps) if len(train_loader) else 0
        scheduler, per_batch = create_scheduler(
            optimizer,
            final_scheduler_choice,
            args.final_train_epochs,
            effective_steps_per_epoch,
            final_lr,
        )

        history: list[dict[str, Any]] = []
        optimizer_steps_total = 0
        ema_updates_total = 0
        swa_updates_total = 0
        memory_sweeps_total = 0
        memory_sweep_time_total = 0.0
        memory_sweep_last_free_final = float("nan")
        memory_sweep_last_reserved_final = float("nan")
        best_state: dict[str, Any] | None = None
        best_entry: dict[str, Any] | None = None
        best_model_source = "model"
        best_accuracy = -float("inf")
        epochs_since_improvement = 0
        total_epochs = 0
        train_loss = 0.0
        train_acc = 0.0
        eval_loss = 0.0
        eval_acc = 0.0
        current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
        final_stage_distillation = (
            distillation_config if args.distill_keep_during_final and distillation_config is not None else None
        )
        distill_epochs_completed = 0
        if args.distill_epochs > 0 and distillation_config is not None and distillation_logits is not None:
            distill_dataset = IntentDataset(
                final_texts,
                final_labels,
                vocab=vocab,
                vocab_config=vocab_config,
                label_to_idx=label_to_idx,
                max_len=max_seq_len,
                sample_weights=distillation_weights if distillation_weights is not None else final_weights,
                tokenizer=tokenizer_obj,
                tokenizer_cache=tokenizer_cache,
                embedding_model=embedding_fn,
                teacher_logits=distillation_logits,
                emotion_vectors=final_emotion_vectors if (emotion_enabled and emotion_dim > 0) else None,
                emotion_dim=emotion_dim if emotion_enabled else 0,
                pin_memory=dataset_pin_memory,
                target_device=dataset_embedding_target_device,
            )
            distill_loader: DataLoader[BatchExample] = create_data_loader(
                distill_dataset,
                batch_size=final_batch_size,
                shuffle=True,
            )
            distill_keyword_vectors: list[list[float]] | None = None
            if final_keyword_calibrator is not None or final_cognitive_router is not None:
                distill_keyword_vectors = [
                    compose_logit_adjustments(
                        text,
                        calibrator=final_keyword_calibrator,
                        router=final_cognitive_router,
                    )
                    for text in final_texts
                ]
            distill_eval_dataset = IntentDataset(
                final_texts,
                final_labels,
                vocab=vocab,
                vocab_config=vocab_config,
                label_to_idx=label_to_idx,
                max_len=max_seq_len,
                sample_weights=distillation_weights if distillation_weights is not None else final_weights,
                tokenizer=tokenizer_obj,
                tokenizer_cache=tokenizer_cache,
                embedding_model=embedding_fn,
                teacher_logits=distillation_logits,
                emotion_vectors=final_emotion_vectors if (emotion_enabled and emotion_dim > 0) else None,
                emotion_dim=emotion_dim if emotion_enabled else 0,
                keyword_vectors=distill_keyword_vectors,
                pin_memory=dataset_pin_memory,
                target_device=dataset_embedding_target_device,
            )
            distill_eval_loader: DataLoader[BatchExample] = create_data_loader(
                distill_eval_dataset,
                batch_size=final_batch_size,
            )
            for _ in range(1, args.distill_epochs + 1):
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
                    amp_enabled=use_amp,
                    amp_device_type=amp_device_type,
                    max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
                    distillation_config=distillation_config,
                    emotion_config=final_emotion_config,
                    meta_config=final_meta_config,
                    neuro_config=final_neuro_config,
                    discovery_config=final_discovery_config,
                    transcendent_config=final_transcendent_config,
                    frontier_config=final_frontier_config,
                    rdrop_config=final_rdrop_config,
                    collect_performance_stats=speed_logger.enabled,
                    memory_regulator=vram_regulator,
                    micro_batch_size=micro_batch_size,
                    memory_sweeper=cuda_memory_sweeper,
                )
                optimizer_steps_total += stats["optimizer_steps"]
                ema_updates_total += stats["ema_updates"]
                swa_updates_total += stats["swa_updates"]
                memory_sweeps_total += int(stats.get("memory_sweeps", 0))
                memory_sweep_time_total += float(stats.get("memory_sweep_time", 0.0))
                last_free = stats.get("memory_sweep_last_free")
                if isinstance(last_free, float) and math.isfinite(last_free):
                    memory_sweep_last_free_final = last_free
                last_reserved = stats.get("memory_sweep_last_reserved")
                if isinstance(last_reserved, float) and math.isfinite(last_reserved):
                    memory_sweep_last_reserved_final = last_reserved
                if speed_logger.enabled:
                    speed_logger.record_training_epoch(
                        stage="final:distill",
                        epoch=total_epochs,
                        seconds=float(stats.get("duration", time.perf_counter() - epoch_start)),
                        examples=float(stats.get("examples", 0.0)),
                        tokens=float(stats.get("tokens", 0.0)),
                        batches=float(stats.get("batches", 0.0)),
                        passes=float(stats.get("examples", 0.0)),
                    )

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

                eval_context: contextlib.AbstractContextManager[None]
                if eval_source == "ema" and hasattr(eval_model, "activate"):
                    eval_context = eval_model.activate(device)  # type: ignore[attr-defined]
                else:
                    eval_context = contextlib.nullcontext()
                with eval_context:
                    eval_loss, eval_acc, eval_targets, eval_predictions, eval_probabilities = cast(
                        tuple[float, float, list[int], list[int], list[list[float]]],
                        evaluate(
                            eval_model,
                            distill_eval_loader,
                            criterion,
                            device,
                            return_details=True,
                            emotion_config=final_emotion_config,
                            meta_stacker=final_meta_stacker,
                        ),
                    )
                final_curriculum_summary: dict[str, Any] | None = None
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
                        "rdrop_loss": float(stats.get("rdrop_loss", 0.0)),
                        "rdrop_kl": float(stats.get("rdrop_kl", 0.0)),
                        "rdrop_passes": float(stats.get("rdrop_passes", 0.0)),
                        "rdrop_alpha": float(stats.get("rdrop_alpha", 0.0)),
                        "frontier_loss": float(stats.get("frontier_loss", 0.0)),
                        "frontier_novelty": float(stats.get("frontier_novelty", 0.0)),
                        "frontier_abstraction": float(stats.get("frontier_abstraction", 0.0)),
                        "frontier_transfer": float(stats.get("frontier_transfer", 0.0)),
                        "frontier_curiosity": float(stats.get("frontier_curiosity", 0.0)),
                        "frontier_emotion": float(stats.get("frontier_emotion", 0.0)),
                        "frontier_meta": float(stats.get("frontier_meta", 0.0)),
                        "frontier_diversity": float(stats.get("frontier_diversity", 0.0)),
                        "frontier_updates": float(stats.get("frontier_updates", 0.0)),
                        "moe_loss": float(stats.get("moe_loss", 0.0)),
                        "moe_entropy": float(stats.get("moe_entropy", 0.0)),
                        "moe_entropy_gap": float(stats.get("moe_entropy_gap", 0.0)),
                        "moe_balance": float(stats.get("moe_balance", 0.0)),
                        "moe_active": float(stats.get("moe_active", 0.0)),
                        "moe_max_gate": float(stats.get("moe_max_gate", 0.0)),
                        "moe_batches": float(stats.get("moe_batches", 0.0)),
                        "moe_utilisation_mean": float(stats.get("moe_utilisation_mean", 0.0)),
                        "moe_utilisation_min": float(stats.get("moe_utilisation_min", 0.0)),
                        "moe_utilisation_max": float(stats.get("moe_utilisation_max", 0.0)),
                    }
                )
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"Final stage distill epoch {total_epochs:03d} "
                    f"train_loss={train_loss:.4f} val_loss={eval_loss:.4f} "
                    f"train_acc={train_acc * 100:.2f}% val_acc={eval_acc * 100:.2f}% "
                    f"lr={current_lr:.6f} ({elapsed:.1f}s)"
                )
                if stats.get("rdrop_loss", 0.0):
                    print(
                        "   -> r-drop kl "
                        f"{stats.get('rdrop_kl', 0.0):.4f} "
                        f"loss {stats.get('rdrop_loss', 0.0):.4f} "
                        f"passes {int(stats.get('rdrop_passes', 0.0))}"
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
                        f"   -> curriculum avg×{final_curriculum_summary['avg_multiplier']:.2f} "
                        f"(boosted {final_curriculum_summary['boosted']}, dampened {final_curriculum_summary['dampened']}, "
                        f"examples {final_curriculum_summary['examples']}); hardest {preview}"
                    )
                if final_meta_config is not None and final_meta_config.enabled:
                    print(
                        "   -> meta-introspection loss "
                        f"{stats.get('meta_loss', 0.0):.4f} "
                        f"gap {stats.get('meta_gap', 0.0):.3f} "
                        f"coverage {stats.get('meta_coverage', 0.0):.2f}"
                    )
                if final_neuro_config is not None and final_neuro_config.enabled:
                    print(
                        "   -> neuro-symbolic loss "
                        f"{stats.get('neuro_loss', 0.0):.4f} "
                        f"struct {stats.get('neuro_structural', 0.0):.4f} "
                        f"cohesion {stats.get('neuro_cohesion', 0.0):.3f} "
                        f"entropy {stats.get('neuro_entropy', 0.0):.3f}"
                    )
                if final_discovery_config is not None and final_discovery_config.enabled:
                    print(
                        "   -> self-discovery loss "
                        f"{stats.get('discovery_loss', 0.0):.4f} "
                        f"align {stats.get('discovery_alignment', 0.0):.4f} "
                        f"curiosity {stats.get('discovery_curiosity', 0.0):.3f}"
                    )
                if final_transcendent_config is not None and final_transcendent_config.enabled:
                    print(
                        "   -> transcendent cognition loss "
                        f"{stats.get('transcendent_loss', 0.0):.4f} "
                        f"coherence {stats.get('transcendent_coherence', 0.0):.3f} "
                        f"stability {stats.get('transcendent_stability', 0.0):.4f}"
                    )
                if final_frontier_config is not None and final_frontier_config.enabled:
                    print(
                        "   -> frontier intelligence loss "
                        f"{stats.get('frontier_loss', 0.0):.4f} "
                        f"novelty {stats.get('frontier_novelty', 0.0):.4f} "
                        f"diversity {stats.get('frontier_diversity', 0.0):.3f}"
                    )
                if stats.get("moe_batches", 0.0):
                    print(
                        "   -> mixture-of-experts loss "
                        f"{stats.get('moe_loss', 0.0):.4f} "
                        f"entropy {stats.get('moe_entropy', 0.0):.3f} "
                        f"balance {stats.get('moe_balance', 0.0):.4f} "
                        f"active {stats.get('moe_active', 0.0):.2f} "
                        f"max {stats.get('moe_max_gate', 0.0):.3f}"
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

        for _ in range(1, args.final_train_epochs + 1):
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
                amp_enabled=use_amp,
                amp_device_type=amp_device_type,
                max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
                distillation_config=final_stage_distillation,
                emotion_config=final_emotion_config,
                meta_config=final_meta_config,
                neuro_config=final_neuro_config,
                discovery_config=final_discovery_config,
                transcendent_config=final_transcendent_config,
                frontier_config=final_frontier_config,
                rdrop_config=final_rdrop_config,
                collect_performance_stats=speed_logger.enabled,
                memory_regulator=vram_regulator,
                micro_batch_size=micro_batch_size,
                memory_sweeper=cuda_memory_sweeper,
            )
            optimizer_steps_total += stats["optimizer_steps"]
            ema_updates_total += stats["ema_updates"]
            swa_updates_total += stats["swa_updates"]
            memory_sweeps_total += int(stats.get("memory_sweeps", 0))
            memory_sweep_time_total += float(stats.get("memory_sweep_time", 0.0))
            last_free = stats.get("memory_sweep_last_free")
            if isinstance(last_free, float) and math.isfinite(last_free):
                memory_sweep_last_free_final = last_free
            last_reserved = stats.get("memory_sweep_last_reserved")
            if isinstance(last_reserved, float) and math.isfinite(last_reserved):
                memory_sweep_last_reserved_final = last_reserved
            if speed_logger.enabled:
                speed_logger.record_training_epoch(
                    stage="final:supervised",
                    epoch=total_epochs,
                    seconds=float(stats.get("duration", time.perf_counter() - epoch_start)),
                    examples=float(stats.get("examples", 0.0)),
                    tokens=float(stats.get("tokens", 0.0)),
                    batches=float(stats.get("batches", 0.0)),
                    passes=float(stats.get("examples", 0.0)),
                )

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

            eval_context: contextlib.AbstractContextManager[None]
            if eval_source == "ema" and hasattr(eval_model, "activate"):
                eval_context = eval_model.activate(device)  # type: ignore[attr-defined]
            else:
                eval_context = contextlib.nullcontext()
            with eval_context:
                eval_loss, eval_acc, eval_targets, eval_predictions, eval_probabilities = cast(
                    tuple[float, float, list[int], list[int], list[list[float]]],
                    evaluate(
                        eval_model,
                        eval_loader,
                        criterion,
                        device,
                        return_details=True,
                        emotion_config=final_emotion_config,
                        meta_stacker=final_meta_stacker,
                    ),
                )
            eval_metrics = compute_classification_metrics(
                eval_targets,
                eval_predictions,
                label_to_idx=label_to_idx,
            )
            current_lr = optimizer.param_groups[0]["lr"]
            final_curriculum_summary: dict[str, Any] | None = None
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
                    "class_metrics": eval_metrics,
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
                    "rdrop_loss": float(stats.get("rdrop_loss", 0.0)),
                    "rdrop_kl": float(stats.get("rdrop_kl", 0.0)),
                    "rdrop_passes": float(stats.get("rdrop_passes", 0.0)),
                    "rdrop_alpha": float(stats.get("rdrop_alpha", 0.0)),
                    "frontier_loss": float(stats.get("frontier_loss", 0.0)),
                    "frontier_novelty": float(stats.get("frontier_novelty", 0.0)),
                    "frontier_abstraction": float(stats.get("frontier_abstraction", 0.0)),
                    "frontier_transfer": float(stats.get("frontier_transfer", 0.0)),
                    "frontier_curiosity": float(stats.get("frontier_curiosity", 0.0)),
                    "frontier_emotion": float(stats.get("frontier_emotion", 0.0)),
                    "frontier_meta": float(stats.get("frontier_meta", 0.0)),
                    "frontier_diversity": float(stats.get("frontier_diversity", 0.0)),
                    "frontier_updates": float(stats.get("frontier_updates", 0.0)),
                    "moe_loss": float(stats.get("moe_loss", 0.0)),
                    "moe_entropy": float(stats.get("moe_entropy", 0.0)),
                    "moe_entropy_gap": float(stats.get("moe_entropy_gap", 0.0)),
                    "moe_balance": float(stats.get("moe_balance", 0.0)),
                    "moe_active": float(stats.get("moe_active", 0.0)),
                    "moe_max_gate": float(stats.get("moe_max_gate", 0.0)),
                    "moe_batches": float(stats.get("moe_batches", 0.0)),
                    "moe_utilisation_mean": float(stats.get("moe_utilisation_mean", 0.0)),
                    "moe_utilisation_min": float(stats.get("moe_utilisation_min", 0.0)),
                    "moe_utilisation_max": float(stats.get("moe_utilisation_max", 0.0)),
                }
            )
            elapsed = time.perf_counter() - epoch_start
            print(
                f"Final training epoch {total_epochs:03d} "
                f"train_loss={train_loss:.4f} val_loss={eval_loss:.4f} "
                f"train_acc={train_acc * 100:.2f}% val_acc={eval_acc * 100:.2f}% "
                f"lr={current_lr:.6f} ({elapsed:.1f}s)"
            )
            if stats.get("rdrop_loss", 0.0):
                print(
                    "   -> r-drop kl "
                    f"{stats.get('rdrop_kl', 0.0):.4f} "
                    f"loss {stats.get('rdrop_loss', 0.0):.4f} "
                    f"passes {int(stats.get('rdrop_passes', 0.0))}"
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
                    f"   -> curriculum avg×{final_curriculum_summary['avg_multiplier']:.2f} "
                    f"(boosted {final_curriculum_summary['boosted']}, dampened {final_curriculum_summary['dampened']}, "
                    f"examples {final_curriculum_summary['examples']}); hardest {preview}"
                )
            if final_meta_config is not None and final_meta_config.enabled:
                print(
                    "   -> meta-introspection loss "
                    f"{stats.get('meta_loss', 0.0):.4f} "
                    f"gap {stats.get('meta_gap', 0.0):.3f} "
                    f"coverage {stats.get('meta_coverage', 0.0):.2f}"
                )
            if final_neuro_config is not None and final_neuro_config.enabled:
                print(
                    "   -> neuro-symbolic loss "
                    f"{stats.get('neuro_loss', 0.0):.4f} "
                    f"struct {stats.get('neuro_structural', 0.0):.4f} "
                    f"cohesion {stats.get('neuro_cohesion', 0.0):.3f} "
                    f"entropy {stats.get('neuro_entropy', 0.0):.3f}"
                )
            if final_discovery_config is not None and final_discovery_config.enabled:
                print(
                    "   -> self-discovery loss "
                    f"{stats.get('discovery_loss', 0.0):.4f} "
                    f"align {stats.get('discovery_alignment', 0.0):.4f} "
                    f"curiosity {stats.get('discovery_curiosity', 0.0):.3f}"
                )
            if final_transcendent_config is not None and final_transcendent_config.enabled:
                print(
                    "   -> transcendent cognition loss "
                    f"{stats.get('transcendent_loss', 0.0):.4f} "
                    f"coherence {stats.get('transcendent_coherence', 0.0):.3f} "
                    f"stability {stats.get('transcendent_stability', 0.0):.4f}"
                )
            if final_frontier_config is not None and final_frontier_config.enabled:
                print(
                    "   -> frontier intelligence loss "
                    f"{stats.get('frontier_loss', 0.0):.4f} "
                    f"novelty {stats.get('frontier_novelty', 0.0):.4f} "
                    f"diversity {stats.get('frontier_diversity', 0.0):.3f}"
                )
            if stats.get("moe_batches", 0.0):
                print(
                    "   -> mixture-of-experts loss "
                    f"{stats.get('moe_loss', 0.0):.4f} "
                    f"entropy {stats.get('moe_entropy', 0.0):.3f} "
                    f"balance {stats.get('moe_balance', 0.0):.4f} "
                    f"active {stats.get('moe_active', 0.0):.2f} "
                    f"max {stats.get('moe_max_gate', 0.0):.3f}"
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

        reference_model: nn.Module = final_model
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
        assert best_entry is not None

        distillation_stats.setdefault("teacher_indices", distillation_stats.get("teacher_indices", []))
        distillation_stats.setdefault("teacher_pool_mean_accuracy", float(distillation_stats.get("teacher_pool_mean_accuracy", 0.0)))
        distillation_stats.setdefault("teacher_coverage", 0.0)
        distillation_stats.setdefault("teacher_active_examples", 0)
        distillation_stats["epochs_completed"] = int(distill_epochs_completed)
        distillation_stats["applied_during_final"] = bool(final_stage_distillation is not None)

        final_model.load_state_dict(best_state)
        final_model.to(device)

        if args.meta_stacker:
            candidate_stacker = MetaIntentStacker(
                label_to_idx=label_to_idx,
                scale=args.meta_stacker_scale,
                regularization=args.meta_stacker_regularization,
                max_iter=args.meta_stacker_max_iter,
                min_accuracy=args.meta_stacker_min_accuracy,
            )
            trained = candidate_stacker.fit_from_dataloader(
                final_model,
                eval_loader,
                device,
                emotion_config=final_emotion_config,
            )
            final_meta_metadata = candidate_stacker.export_metadata()
            if trained:
                final_meta_stacker = candidate_stacker

        final_loss, final_acc, final_targets, final_predictions, _ = cast(
            tuple[float, float, list[int], list[int], list[list[float]]],
            evaluate(
                final_model,
                eval_loader,
                criterion,
                device,
                return_details=True,
                emotion_config=final_emotion_config,
                meta_stacker=final_meta_stacker,
            ),
        )
        final_metrics = compute_classification_metrics(
            final_targets,
            final_predictions,
            label_to_idx=label_to_idx,
        )

        final_meta_snapshot: dict[str, Any] | None = None
        if final_meta_config is not None and final_meta_config.enabled:
            final_meta_snapshot = final_meta_config.introspector.snapshot()

        final_micro_batch_values = [
            float(entry.get("micro_batch_size", 0.0))
            for entry in history
            if entry.get("micro_batch_size")
        ]
        final_micro_batches_values = [
            float(entry.get("micro_batches", 0.0))
            for entry in history
            if entry.get("micro_batches")
        ]
        micro_size_mean = (
            sum(final_micro_batch_values) / len(final_micro_batch_values)
            if final_micro_batch_values
            else 0.0
        )
        micro_size_last = final_micro_batch_values[-1] if final_micro_batch_values else 0.0
        micro_batches_mean = (
            sum(final_micro_batches_values) / len(final_micro_batches_values)
            if final_micro_batches_values
            else 0.0
        )
        micro_batches_last = (
            final_micro_batches_values[-1] if final_micro_batches_values else 0.0
        )

        metadata = {
            "stage": "final_full",
            "base_fold_accuracy": best_fold.val_accuracy,
            "trainer_version": TRAINER_VERSION,
            "memory_guard": _memory_guard_summary(args),
            "cuda_memory_guard": _cuda_memory_guard_summary(args),
            "performance_overdrive": _performance_overdrive_summary(args),
            "model_name": args.model_name,
            "overdrive_profile": bool(args.overdrive_profile),
            "encoder_type": args.encoder_type,
            "transformer_model": args.transformer_model if args.encoder_type == "transformer" else None,
            "transformer_layerwise_decay": (
                args.transformer_layerwise_decay if args.encoder_type == "transformer" else None
            ),
            "rdrop_alpha": float(args.rdrop_alpha),
            "rdrop_forward_passes": int(args.rdrop_forward_passes),
            "dataset_path": str(args.dataset),
            "dataset_checksum": dataset_checksum,
            "dataset_examples": len(final_texts),
            "num_labels": len(label_to_idx),
            "colab_environment": (
                dict(getattr(args, "colab_adjustments", {}))
                if getattr(args, "colab_environment", False)
                else None
            ),
            "vocab_settings": {
                "include_bigrams": bool(vocab_config.include_bigrams),
                "include_trigrams": bool(vocab_config.include_trigrams),
                "include_char_ngrams": bool(vocab_config.include_char_ngrams),
                "char_ngram_min": int(vocab_config.char_ngram_min),
                "char_ngram_max": int(vocab_config.char_ngram_max),
                "char_ngram_limit": int(vocab_config.char_ngram_limit),
                "max_size": int(args.vocab_max_size) if args.vocab_max_size > 0 else None,
                "extra_sources": sorted(active_vocab_sources),
                "extra_fragments": len(deduped_extra_texts) if deduped_extra_texts is not None else 0,
            },
            "label_to_idx": label_to_idx,
            "max_seq_len": max_seq_len,
            "metadata_feature_strategy": args.metadata_feature_strategy,
            "metadata_feature_dim": metadata_dim,
            "metadata_fields": metadata_encoder.export() if metadata_encoder is not None else None,
            "lexicon_feature_dim": lexicon_dim,
            "emotion_feature_dim": emotion_dim,
            "bilstm_conv_head": bool(args.bilstm_conv_head if args.encoder_type == "bilstm" else False),
            "bilstm_conv_kernels": bilstm_conv_kernel_sizes if args.encoder_type == "bilstm" else [],
            "bilstm_conv_channels": (
                int(args.bilstm_conv_channels) if args.encoder_type == "bilstm" else 0
            ),
            "bilstm_conv_dropout": (
                float(args.bilstm_conv_dropout) if args.encoder_type == "bilstm" else 0.0
            ),
            "keyword_calibration": final_keyword_metadata,
            "meta_stacker": final_meta_metadata,
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
            "micro_batch_history": {
                "size_mean": micro_size_mean,
                "size_last": micro_size_last,
                "batches_mean": micro_batches_mean,
                "batches_last": micro_batches_last,
            },
            "advanced_training": {
                "gradient_accumulation_steps": grad_acc_steps,
                "ema_decay": args.ema_decay,
                "ema_start_epoch": args.ema_start_epoch,
                "ema_updates": ema_updates_total,
                "ema_used_for_evaluation": args.ema_use_for_eval,
                "ema_offload_cpu": bool(args.ema_offload_cpu and using_cuda),
                "ema_offload_dtype": args.ema_offload_dtype,
                "swa_start_epoch": args.swa_start_epoch,
                "swa_lr": args.swa_lr if args.swa_lr > 0 else final_lr,
                "swa_anneal_epochs": args.swa_anneal_epochs,
                "swa_updates": swa_updates_total,
                "optimizer_steps": optimizer_steps_total,
                "memory_sweeps": memory_sweeps_total,
                "memory_sweep_time": memory_sweep_time_total,
                "memory_sweep_last_free": memory_sweep_last_free_final,
                "memory_sweep_last_reserved": memory_sweep_last_reserved_final,
                "best_model_source": best_model_source,
                "learning_rate": final_lr,
                "weight_decay": final_weight_decay,
                "scheduler": final_scheduler_choice,
                "rdrop_alpha": float(args.rdrop_alpha),
                "rdrop_forward_passes": int(args.rdrop_forward_passes),
                "transformer_layerwise_decay": (
                    float(args.transformer_layerwise_decay)
                    if args.encoder_type == "transformer"
                    else None
                ),
                "micro_batch_size": int(args.micro_batch_size_effective),
                "micro_batch_regulator": vram_regulator.export() if vram_regulator is not None else None,
                "cuda_memory_sweeper": (
                    cuda_memory_sweeper.export() if cuda_memory_sweeper is not None else None
                ),
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
            metadata["sentence_transformer_mlp_layers"] = args.st_mlp_layers
            metadata["sentence_transformer_mlp_expansion"] = args.st_mlp_expansion
            metadata["sentence_transformer_hidden_dims"] = list(args.st_mlp_hidden_dims)
            metadata["sentence_transformer_activation"] = args.st_mlp_activation
            metadata["sentence_transformer_final_dropout"] = args.st_final_dropout
            metadata["sentence_transformer_layer_norm"] = bool(args.st_mlp_layer_norm)
            metadata["sentence_transformer_residual"] = bool(args.st_mlp_residual)

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
        final_moe_entries = [entry for entry in history if entry.get("moe_batches", 0.0)]
        final_moe_loss_values = [
            float(entry.get("moe_loss", 0.0)) for entry in final_moe_entries
        ]
        final_moe_entropy_values = [
            float(entry.get("moe_entropy", 0.0)) for entry in final_moe_entries
        ]
        final_moe_gap_values = [
            float(entry.get("moe_entropy_gap", 0.0)) for entry in final_moe_entries
        ]
        final_moe_balance_values = [
            float(entry.get("moe_balance", 0.0)) for entry in final_moe_entries
        ]
        final_moe_active_values = [
            float(entry.get("moe_active", 0.0)) for entry in final_moe_entries
        ]
        final_moe_max_values = [
            float(entry.get("moe_max_gate", 0.0)) for entry in final_moe_entries
        ]
        final_moe_util_mean_values = [
            float(entry.get("moe_utilisation_mean", 0.0))
            for entry in final_moe_entries
        ]
        final_moe_util_min_values = [
            float(entry.get("moe_utilisation_min", 0.0))
            for entry in final_moe_entries
        ]
        final_moe_util_max_values = [
            float(entry.get("moe_utilisation_max", 0.0))
            for entry in final_moe_entries
        ]
        final_moe_summary: dict[str, Any] = {}
        if final_moe_loss_values:
            final_moe_summary["moe_loss_mean"] = float(
                sum(final_moe_loss_values) / len(final_moe_loss_values)
            )
            final_moe_summary["moe_loss_last"] = float(final_moe_loss_values[-1])
        if final_moe_entropy_values:
            final_moe_summary["moe_entropy_mean"] = float(
                sum(final_moe_entropy_values) / len(final_moe_entropy_values)
            )
            final_moe_summary["moe_entropy_last"] = float(final_moe_entropy_values[-1])
        if final_moe_gap_values:
            final_moe_summary["moe_entropy_gap_mean"] = float(
                sum(final_moe_gap_values) / len(final_moe_gap_values)
            )
            final_moe_summary["moe_entropy_gap_last"] = float(final_moe_gap_values[-1])
        if final_moe_balance_values:
            final_moe_summary["moe_balance_mean"] = float(
                sum(final_moe_balance_values) / len(final_moe_balance_values)
            )
            final_moe_summary["moe_balance_last"] = float(final_moe_balance_values[-1])
        if final_moe_active_values:
            final_moe_summary["moe_active_mean"] = float(
                sum(final_moe_active_values) / len(final_moe_active_values)
            )
            final_moe_summary["moe_active_last"] = float(final_moe_active_values[-1])
        if final_moe_max_values:
            final_moe_summary["moe_max_gate_mean"] = float(
                sum(final_moe_max_values) / len(final_moe_max_values)
            )
            final_moe_summary["moe_max_gate_last"] = float(final_moe_max_values[-1])
        if final_moe_util_mean_values:
            final_moe_summary["moe_utilisation_mean_mean"] = float(
                sum(final_moe_util_mean_values) / len(final_moe_util_mean_values)
            )
            final_moe_summary["moe_utilisation_mean_last"] = float(
                final_moe_util_mean_values[-1]
            )
        if final_moe_util_min_values:
            final_moe_summary["moe_utilisation_min_mean"] = float(
                sum(final_moe_util_min_values) / len(final_moe_util_min_values)
            )
            final_moe_summary["moe_utilisation_min_last"] = float(
                final_moe_util_min_values[-1]
            )
        if final_moe_util_max_values:
            final_moe_summary["moe_utilisation_max_mean"] = float(
                sum(final_moe_util_max_values) / len(final_moe_util_max_values)
            )
            final_moe_summary["moe_utilisation_max_last"] = float(
                final_moe_util_max_values[-1]
            )
        if final_moe_entries:
            final_moe_summary["moe_batches_last"] = float(
                final_moe_entries[-1].get("moe_batches", 0.0)
            )
        if final_moe_summary:
            metadata.update(final_moe_summary)
        metrics = {
            "model_name": args.model_name,
            "trainer_version": TRAINER_VERSION,
            "memory_guard": _memory_guard_summary(args),
            "cuda_memory_guard": _cuda_memory_guard_summary(args),
            "performance_overdrive": _performance_overdrive_summary(args),
            "timestamp_utc": datetime.now(UTC_ZONE).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "overdrive_profile": bool(args.overdrive_profile),
            "dataset_examples": len(final_texts),
            "dataset_checksum": dataset_checksum,
            "num_labels": len(label_to_idx),
            "encoder_type": args.encoder_type,
            "metadata_feature_strategy": args.metadata_feature_strategy,
            "metadata_feature_dim": metadata_dim,
            "lexicon_feature_dim": lexicon_dim,
            "training_accuracy": float(final_acc),
            "training_loss": float(final_loss),
            "training_macro_f1": float(final_metrics["macro_f1"]),
            "training_weighted_f1": float(final_metrics["weighted_f1"]),
            "epochs_ran": float(len(history)),
            "micro_batch_size": float(args.micro_batch_size_effective),
            "micro_batch_size_mean": micro_size_mean,
            "micro_batch_size_last": micro_size_last,
            "micro_batches_mean": micro_batches_mean,
            "micro_batches_last": micro_batches_last,
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
            "keyword_calibration_enabled": bool(final_keyword_calibrator is not None),
            "keyword_calibration_feature_count": int(
                final_keyword_calibrator.feature_count if final_keyword_calibrator is not None else 0
            ),
            "cognitive_router_enabled": bool(final_cognitive_router is not None),
            "cognitive_router_trigger_total": int(
                final_cognitive_router.total_triggers if final_cognitive_router is not None else 0
            ),
            "cognitive_router_examples": int(
                final_cognitive_router.adjusted_examples if final_cognitive_router is not None else 0
            ),
            "meta_stacker_enabled": bool(final_meta_stacker is not None),
            "meta_stacker_training_accuracy": float(
                final_meta_metadata.get("training_accuracy", 0.0)
            ),
            "meta_stacker_trained_samples": int(final_meta_metadata.get("trained_samples", 0)),
            "meta_stacker_feature_count": int(final_meta_metadata.get("feature_count", 0)),
        }
        if vram_regulator is not None:
            metrics["micro_batch_regulator"] = vram_regulator.export()
        if cuda_memory_sweeper is not None:
            metrics["cuda_memory_sweeper"] = cuda_memory_sweeper.export()
        if auto_actions:
            metrics["auto_optimizations"] = list(auto_actions)
        if final_moe_summary:
            metrics.update(final_moe_summary)
        metrics["emotion_reasoner_enabled"] = bool(emotion_enabled and emotion_dim > 0)
        metrics["hardware_monitor"] = hardware_monitor.snapshot()
        if overdrive_simulation_summary:
            metrics["overdrive_simulation"] = overdrive_simulation_summary
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
            metrics["transformer_layerwise_decay"] = float(args.transformer_layerwise_decay)
            metrics["rdrop_alpha"] = float(args.rdrop_alpha)
            metrics["rdrop_forward_passes"] = int(args.rdrop_forward_passes)
            if tokenizer_obj is not None:
                metrics["tokenizer_vocab_size"] = len(tokenizer_obj)
        if args.encoder_type == "st":
            metrics["sentence_transformer_model"] = args.sentence_transformer_model
            metrics["sentence_transformer_hidden_dim"] = args.st_hidden_dim
            metrics["sentence_transformer_dropout"] = args.st_dropout
            metrics["sentence_transformer_mlp_layers"] = args.st_mlp_layers
            metrics["sentence_transformer_mlp_expansion"] = args.st_mlp_expansion
            metrics["sentence_transformer_hidden_dims"] = list(args.st_mlp_hidden_dims)
            metrics["sentence_transformer_activation"] = args.st_mlp_activation
            metrics["sentence_transformer_final_dropout"] = args.st_final_dropout
            metrics["sentence_transformer_layer_norm"] = bool(args.st_mlp_layer_norm)
            metrics["sentence_transformer_residual"] = bool(args.st_mlp_residual)
        if final_curriculum_manager is not None:
            metrics.update(final_curriculum_manager.export_metrics())
        elif args.adaptive_curriculum:
            metrics.setdefault("curriculum_updates", 0)
            metrics.setdefault("curriculum_avg_multiplier", 1.0)
            metrics.setdefault("curriculum_max_multiplier", 1.0)
            metrics.setdefault("curriculum_min_multiplier", 1.0)

        evaluation_outputs: list[dict[str, Any]] = []
        for sample in evaluation_inputs:
            analysis_features = inspect_text_characteristics(sample)
            prediction = predict_with_trace(
                final_model,
                sample,
                vocab=vocab,
                label_to_idx=label_to_idx,
                max_len=max_seq_len,
                device=device,
                tokenizer=tokenizer_obj,
                tokenizer_cache=tokenizer_cache,
                embedding_model=embedding_fn,
                vocab_config=vocab_config,
                emotion_encoder=emotion_lexicon if (emotion_enabled and emotion_dim > 0) else None,
                emotion_dim=emotion_dim,
                emotion_config=final_emotion_config,
                metadata=None,
                metadata_encoder=metadata_encoder if metadata_dim > 0 else None,
                lexicon_dim=lexicon_dim,
                metadata_dim=metadata_dim,
                calibrator=final_keyword_calibrator,
                symbolic_router=final_cognitive_router,
                meta_stacker=final_meta_stacker,
            )
            response = generate_response(prediction.label, sample)
            valuation_summary: dict[str, Any] = {
                "question_type": analysis_features.get("question_type"),
                "false_question": bool(analysis_features.get("false_question")),
                "seduction_style": analysis_features.get("seduction_style"),
                "comma_count": int(analysis_features.get("comma_count", 0)),
                "uppercase_ratio": round(float(analysis_features.get("uppercase_ratio", 0.0)), 3),
            }
            uppercase_tokens = list(analysis_features.get("uppercase_tokens", []))
            if uppercase_tokens:
                valuation_summary["uppercase_tokens"] = uppercase_tokens[:3]
            seduction_terms = list(analysis_features.get("seduction_terms", []))
            if seduction_terms:
                valuation_summary["seduction_terms"] = sorted(set(seduction_terms))[:3]
            valuation_rng = _orion_seed_rng(f"valuation::final::{sample}")
            valuation_reflections = craft_orion_reflections(
                analysis_features,
                label=prediction.label,
                rng=valuation_rng,
                context="valuation",
            )[:3]
            entry: dict[str, Any] = {
                "input": sample,
                "predicted_intent": prediction.label,
                "confidence": prediction.confidence,
                "top_intents": [
                    {"label": label, "confidence": score}
                    for label, score in prediction.top_predictions
                ],
                "response": response.message,
                "response_strategy": response.strategy,
                "valuation": valuation_summary,
            }
            if response.basis:
                entry["response_basis"] = response.basis
            if valuation_reflections:
                entry["valuation_reflections"] = valuation_reflections
            evaluation_outputs.append(entry)

        if final_cognitive_router is not None and final_cognitive_router.total_triggers > 0:
            print(
                "Full-data training: cognitive router emitted "
                f"{final_cognitive_router.total_triggers} adjustments across "
                f"{final_cognitive_router.adjusted_examples} analysed texts."
            )

        final_model.cpu()
        if using_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "metadata": metadata,
            "metrics": metrics,
            "model_state": best_state,
            "evaluation_outputs": evaluation_outputs,
            "training_history": history,
            "accuracy": float(final_acc),
            "auto_optimizations": list(auto_actions),
        }

    fold_results: list[FoldResult] = []
    cv_timer = speed_logger.marker()
    fold_passes_total = 0.0
    for fold_idx, (train_indices, val_indices) in enumerate(fold_pairs, start=1):
        fold_timer = speed_logger.marker()
        fold_result = run_single_split(fold_idx, train_indices, val_indices)
        fold_results.append(fold_result)
        fold_examples_raw = (
            fold_result.metadata.get("training_examples_supervised")
            or fold_result.metadata.get("training_examples_final")
            or 0
        )
        try:
            fold_examples = int(fold_examples_raw)
        except (TypeError, ValueError):
            fold_examples = 0
        fold_epochs_raw = fold_result.metrics.get("epochs_ran")
        try:
            fold_epochs = float(fold_epochs_raw) if fold_epochs_raw is not None else 0.0
        except (TypeError, ValueError):
            fold_epochs = 0.0
        fold_passes = float(fold_examples) * max(fold_epochs, 0.0)
        fold_passes_total += fold_passes
        speed_logger.record_fold(
            fold_result.fold_index,
            fold_timer,
            examples=fold_examples,
            epochs=fold_epochs,
            passes=fold_passes,
            add_to_total=not speed_logger.using_epoch_details(),
        )
        gc.collect()
        if using_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

    speed_logger.record_section(
        "cross_validation",
        cv_timer,
        count=len(fold_results),
        passes=fold_passes_total,
        add_to_total=False,
    )

    if not fold_results:
        raise RuntimeError("No folds were produced for training.")

    fold_accuracies = [result.val_accuracy for result in fold_results]
    cv_mean = mean(fold_accuracies)
    cv_std = pstdev(fold_accuracies) if len(fold_accuracies) > 1 else 0.0
    best_fold = max(fold_results, key=lambda result: result.val_accuracy)

    registry = ModelRegistry(args.model_dir, args.model_name, tolerance=args.promotion_tolerance)
    previous_best = registry.best_accuracy()
    promote = registry.is_improvement(best_fold.val_accuracy)

    run_dirs: dict[int, Path] = {}
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

        if auto_actions:
            result.metrics.setdefault("auto_optimizations", list(auto_actions))

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

    final_stage_timer = speed_logger.marker()
    final_stage = run_full_dataset_training(best_fold)
    if final_stage is not None:
        final_metrics_section = final_stage["metrics"]
        final_examples_raw = final_metrics_section.get("dataset_examples", 0)
        try:
            final_examples = int(final_examples_raw)
        except (TypeError, ValueError):
            final_examples = 0
        final_epochs_raw = final_metrics_section.get("epochs_ran")
        try:
            final_epochs = float(final_epochs_raw) if final_epochs_raw is not None else 0.0
        except (TypeError, ValueError):
            final_epochs = 0.0
        final_passes = float(final_examples) * max(final_epochs, 0.0)
        speed_logger.record_section(
            "final_training",
            final_stage_timer,
            passes=final_passes,
            notes={
                "epochs": final_epochs,
                "examples": float(final_examples),
            },
            add_to_total=not speed_logger.using_epoch_details(),
        )
        final_run_tag = "__".join([part for part in [args.run_tag, "final_full"] if part]) or None
        final_run_dir = registry.create_run_directory(final_stage["accuracy"], final_run_tag)
        final_model_to_save = build_model(target_device=torch.device("cpu"))
        final_model_to_save.load_state_dict(final_stage["model_state"])
        final_metadata = final_stage["metadata"]
        final_metrics = final_metrics_section
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
        top_candidates_value = entry.get("top_intents")
        top_candidates: list[dict[str, Any]] = []
        if isinstance(top_candidates_value, Sequence):
            for candidate in top_candidates_value:
                if isinstance(candidate, Mapping):
                    typed_candidate = cast(Mapping[Any, Any], candidate)
                    top_candidates.append(_stringify_mapping(typed_candidate))
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

    speed_logger.finish(total_examples=len(texts))
    if speed_logger.enabled:
        profile = _build_speed_test_profile(
            args,
            average_tokens=average_tokens,
            reference_gflops=_resolve_reference_gflops(args, overdrive_simulation_summary),
            fallback_mode=False,
            observed_per_pass=speed_logger.baseline_per_pass(),
            observed_tokens_per_pass=speed_logger.average_tokens_per_pass(),
        )
        speed_logger.apply_complexity_profile(profile)
        calibration = _build_speed_test_calibration(args)
        if calibration is not None:
            speed_logger.configure_calibration(calibration)
        estimate_target = args.estimate_dataset
        try:
            if estimate_target.resolve(strict=False) == args.dataset.resolve(strict=False):
                estimate_summary = dataset_summary
            else:
                estimate_summary = summarise_labelled_dataset(
                    estimate_target,
                    sample_limit=token_sample_limit if token_sample_limit > 0 else None,
                )
        except OSError:
            estimate_summary = None
        estimated_examples = estimate_summary.examples if estimate_summary is not None else 0
        if estimated_examples > 0 and estimate_summary is not None:
            label = f"Projected runtime for {estimate_target.name}"
            speed_logger.register_estimate(
                label,
                estimated_examples,
                observed_examples=len(texts),
                target_average_tokens=estimate_summary.average_tokens,
                target_total_tokens=estimate_summary.total_tokens,
                observed_average_tokens=dataset_summary.average_tokens,
                observed_total_tokens=dataset_summary.total_tokens,
                dataset_summary=estimate_summary,
                observed_dataset_summary=dataset_summary,
            )
        else:
            print(
                f"Unable to project runtime for {estimate_target}: dataset is missing or empty."
            )

    final_hardware_summary = hardware_monitor.stop_and_summarise()
    args.hardware_monitor_summary = final_hardware_summary
    args.overdrive_simulation_summary = overdrive_simulation_summary
    samples_captured = int(final_hardware_summary.get("samples", 0))
    if final_hardware_summary.get("enabled") and samples_captured > 0:
        power_stats = cast(dict[str, Any], final_hardware_summary.get("power_watts") or {})
        temp_stats = cast(dict[str, Any], final_hardware_summary.get("temperature_c") or {})
        fan_stats = cast(dict[str, Any], final_hardware_summary.get("fan_rpm") or {})
        message_parts = [
            f"Hardware telemetry captured {samples_captured} samples via {final_hardware_summary.get('backend', 'nvidia-smi')}"
        ]
        extras: list[str] = []
        peak_power = power_stats.get("max")
        if peak_power is not None:
            extras.append(f"peak power {peak_power:.1f} W")
        peak_temp = temp_stats.get("max")
        if peak_temp is not None:
            extras.append(f"max temperature {peak_temp:.1f} °C")
        peak_fan = fan_stats.get("max")
        if peak_fan is not None:
            extras.append(f"fan {peak_fan:.0f} RPM")
        if extras:
            message_parts.append("(" + ", ".join(extras) + ")")
        print(" ".join(message_parts) + ".")
    else:
        reason = final_hardware_summary.get("reason") or "telemetry-unavailable"
        print(f"Hardware telemetry summary unavailable ({reason}).")

    speed_logger.report()


if __name__ == "__main__":
    main()
