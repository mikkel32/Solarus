"""Interactive UI for chatting with the latest Orion intent model."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, cast

if TYPE_CHECKING:
    import torch as _torch  # type: ignore[import-not-found]
    torch = _torch
else:  # pragma: no cover - runtime import with optional dependency
    import torch  # type: ignore[import-not-found]
import gradio as gr  # type: ignore[import-not-found]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_intent_classifier import (
    IntentClassifier,
    EmotionallyAdaptiveModel,
    TransformerIntentModel,
    SentenceTransformerClassifier,
    generate_response,
    predict_with_trace,
    VocabularyConfig,
    build_vocab,
    read_dataset,
)
if TYPE_CHECKING:
    TorchModule = _torch.nn.Module
    TorchDevice = _torch.device
    TorchTensor = _torch.Tensor
else:
    torch = cast(Any, torch)
    TorchModule = Any
    TorchDevice = Any
    TorchTensor = Any


TokenizerCache = Callable[[str], Tuple[Sequence[int], Sequence[int]]]
EmbeddingFn = Callable[[str], Sequence[float]]


@dataclass(frozen=True)
class OrionResources:
    model: TorchModule
    device: TorchDevice
    label_to_idx: Dict[str, int]
    vocab: Optional[Dict[str, int]]
    max_seq_len: int
    encoder_type: str
    tokenizer: Optional[object] = None
    tokenizer_cache: Optional[TokenizerCache] = None
    embedding_fn: Optional[EmbeddingFn] = None
    model_dir: Path | None = None


ORION_ROOT = REPO_ROOT / "models"
_DATASET_CACHE: Dict[Path, Tuple[List[str], List[str], List[Dict[str, str]]]] = {}
_UNLABELED_TEXT_CACHE: Dict[Path, List[str]] = {}


def _parse_version(name: str) -> Optional[Tuple[int, ...]]:
    match = re.search(r"orion_v(\d+)(?:[.](\d+))?", name)
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return (major, minor)


def discover_latest_model(models_root: Path = ORION_ROOT) -> Path:
    candidates: List[Tuple[Tuple[int, ...], Path]] = []
    for path in models_root.iterdir():
        if not path.is_dir():
            continue
        version = _parse_version(path.name)
        if version is None:
            continue
        if not (path / "model.pt").exists():
            continue
        candidates.append((version, path))
    if not candidates:
        raise FileNotFoundError(
            f"No Orion checkpoints found under {models_root}."
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _infer_lstm_layers(state_dict: Dict[str, TorchTensor], prefix: str = "") -> int:
    layers: set[int] = set()
    for key in state_dict:
        if prefix:
            if not key.startswith(prefix):
                continue
            lookup = key[len(prefix) :]
        else:
            lookup = key
        if lookup.startswith("lstm.weight_ih_l") and "reverse" not in lookup:
            layer_idx = int(lookup.split("_l")[1].split(".")[0])
            layers.add(layer_idx)
    return max(layers) + 1 if layers else 1


def _infer_conv_head(
    state_dict: Dict[str, TorchTensor], prefix: str = ""
) -> Tuple[List[int], Optional[int]]:
    kernels: set[int] = set()
    conv_channels: Optional[int] = None
    target = f"{prefix}conv_blocks."
    for key, tensor in state_dict.items():
        if not key.startswith(target):
            continue
        suffix = key[len(target) :]
        parts = suffix.split(".")
        if len(parts) >= 3 and parts[1] == "0" and parts[2] == "weight":
            kernel_size = int(tensor.shape[-1])
            kernels.add(kernel_size)
            conv_channels = int(tensor.shape[0])
    return sorted(kernels), conv_channels


def _load_sentence_transformer(
    metadata: Dict[str, object], num_labels: int
) -> Tuple[SentenceTransformerClassifier, Optional[EmbeddingFn]]:
    model_name = metadata.get("sentence_transformer_model")
    hidden_value = metadata.get("sentence_transformer_hidden_dim", 512)
    if isinstance(hidden_value, (int, float, str)):
        hidden_dim = int(hidden_value)
    else:
        hidden_dim = 512
    dropout_value = metadata.get("sentence_transformer_dropout", 0.2)
    if isinstance(dropout_value, (int, float, str)):
        dropout = float(dropout_value)
    else:
        dropout = 0.2
    if model_name is None:
        raise ValueError("Sentence-transformer metadata missing the model name.")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'sentence-transformers' package is required for the ST encoder."
        ) from exc
    st_model = SentenceTransformer(model_name)

    def embedding_fn(text: str) -> Sequence[float]:
        return st_model.encode(text, show_progress_bar=False)

    embedding_dim = len(embedding_fn("placeholder"))
    classifier = SentenceTransformerClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_labels,
        dropout=dropout,
    )
    return classifier, embedding_fn


def _candidate_dataset_paths(dataset_entry: object, *, include_default_dataset: bool = True) -> List[Path]:
    candidates: List[Path] = []
    if dataset_entry:
        primary = Path(str(dataset_entry))
        candidates.append(primary)
        if not primary.is_absolute():
            candidates.append(REPO_ROOT / primary)
        else:
            candidates.append(REPO_ROOT / primary.name)
    if include_default_dataset:
        defaults = [
            REPO_ROOT / "data" / "intent_dataset.csv",
            REPO_ROOT / "data" / "intent_dataset_v1.csv",
        ]
        candidates.extend(defaults)
    unique: List[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path.exists() and path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _load_dataset_cached(path: Path) -> Optional[Tuple[List[str], List[str], List[Dict[str, str]]]]:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if resolved in _DATASET_CACHE:
        return _DATASET_CACHE[resolved]
    if not path.exists():
        return None
    texts, labels, metadata_rows = read_dataset(path)
    _DATASET_CACHE[resolved] = (texts, labels, metadata_rows)
    return texts, labels, metadata_rows


def _load_unlabeled_texts(path: Path) -> List[str]:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if resolved in _UNLABELED_TEXT_CACHE:
        return _UNLABELED_TEXT_CACHE[resolved]
    if not path.exists():
        _UNLABELED_TEXT_CACHE[resolved] = []
        return []
    texts: List[str] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames and "text" in reader.fieldnames:
                for row in reader:
                    value = (row.get("text") or "").strip()
                    if value:
                        texts.append(value)
    except (OSError, UnicodeDecodeError):
        texts = []
    _UNLABELED_TEXT_CACHE[resolved] = texts
    return texts


def _reconstruct_labels_from_dataset(metadata: Dict[str, object], expected_classes: int) -> Optional[Dict[str, int]]:
    dataset_entry = metadata.get("dataset_path")
    candidates = _candidate_dataset_paths(dataset_entry)
    if not candidates:
        return None

    for path in candidates:
        dataset = _load_dataset_cached(path)
        if dataset is None:
            continue
        _, labels, _ = dataset
        ordered = sorted(set(labels))
        if len(ordered) == expected_classes:
            print(f"ℹ️ Reconstructed label mapping from {path}.")
            return {label: idx for idx, label in enumerate(ordered)}
    return None


def _reconstruct_vocab(metadata: Dict[str, object], expected_size: int) -> Optional[Dict[str, int]]:
    dataset_entry = metadata.get("dataset_path")
    candidates = _candidate_dataset_paths(dataset_entry)
    if not candidates:
        return None
    for path in candidates:
        dataset = _load_dataset_cached(path)
        if dataset is None:
            continue
        texts, _, metadata_rows = dataset
        if not texts:
            continue
        metadata_fragments = [value for row in metadata_rows for value in row.values() if value]
        base_extras = list(dict.fromkeys(metadata_fragments)) if metadata_fragments else []
        unlabeled_paths: List[Path] = []
        sibling_unlabeled = [
            path.parent / "unlabeled_pool.csv",
            path.parent / "unlabeled.csv",
        ]
        for sibling in sibling_unlabeled:
            if sibling.exists():
                unlabeled_paths.append(sibling)
        if not unlabeled_paths:
            default_unlabeled = REPO_ROOT / "data" / "unlabeled_pool.csv"
            if default_unlabeled.exists():
                unlabeled_paths.append(default_unlabeled)

        unlabeled_sources: List[Tuple[str, List[str]]] = []
        for unlabeled_path in unlabeled_paths:
            texts_bucket = _load_unlabeled_texts(unlabeled_path)
            if texts_bucket:
                unlabeled_sources.append((unlabeled_path.name, texts_bucket))

        def _build_with_sources(extra_groups: Sequence[Tuple[str, List[str]]]) -> Optional[Dict[str, int]]:
            combined: List[str] = list(base_extras)
            labels: List[str] = ["metadata"] if base_extras else []
            for label, bucket in extra_groups:
                combined.extend(bucket)
                labels.append(label)
            deduped_extra = list(dict.fromkeys(combined)) if combined else None
            vocab_candidate = build_vocab(
                texts,
                min_freq=1,
                config=VocabularyConfig(),
                extra_texts=deduped_extra,
            )
            if len(vocab_candidate) == expected_size:
                label_str = ", ".join(labels) if labels else "dataset"
                print(f"ℹ️ Rebuilt vocabulary from {path} with {len(vocab_candidate)} entries (sources: {label_str}).")
                return vocab_candidate
            return None

        result = _build_with_sources(tuple())
        if result is not None:
            return result
        total_unlabeled = len(unlabeled_sources)
        for mask in range(1, 1 << total_unlabeled):
            selected = [
                unlabeled_sources[idx]
                for idx in range(total_unlabeled)
                if mask & (1 << idx)
            ]
            result = _build_with_sources(selected)
            if result is not None:
                return result
    return None
    return None


def build_orion_resources(model_dir: Optional[Path] = None) -> OrionResources:
    if model_dir is None:
        model_dir = discover_latest_model()
    metadata_path = model_dir / "metadata.json"
    weights_path = model_dir / "model.pt"
    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Orion checkpoint is missing required files in {model_dir}."
        )

    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    raw_label_to_idx = metadata.get("label_to_idx")
    if not isinstance(raw_label_to_idx, dict):
        raise ValueError("Metadata is missing a valid label_to_idx mapping.")
    label_to_idx: Dict[str, int] = {str(label): int(idx) for label, idx in raw_label_to_idx.items()}
    encoder_type = str(metadata.get("encoder_type", "bilstm")).lower()
    max_seq_len = int(metadata.get("max_seq_len", 128))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(weights_path, map_location=device)

    state_keys = list(state_dict.keys())
    has_base_prefix = any(key.startswith("base_model.") for key in state_keys)
    base_prefix = "base_model." if has_base_prefix else ""
    emotion_keys_present = any(
        key.startswith("emotion_transform.") or key.startswith("fusion_gate.") for key in state_keys
    ) or "residual_scale" in state_dict
    is_emotion_adaptive = has_base_prefix and emotion_keys_present

    if has_base_prefix:
        base_keys = [key[len(base_prefix) :] for key in state_keys if key.startswith(base_prefix)]
    else:
        base_keys = state_keys

    def _detect_encoder(default: str) -> str:
        lowered = default.lower()
        if any(k.startswith("model.") for k in base_keys):
            return "transformer"
        if any(k.startswith("embedding.") or k.startswith("lstm.") for k in base_keys):
            return "bilstm"
        if any(k.startswith("output_layer.") for k in base_keys):
            return "st"
        return lowered

    base_encoder_type = _detect_encoder(encoder_type)

    def _prefixed(name: str) -> str:
        return f"{base_prefix}{name}" if base_prefix else name

    def _state_tensor(name: str) -> TorchTensor:
        key = _prefixed(name)
        if key not in state_dict:
            raise KeyError(f"Expected key '{key}' in Orion checkpoint.")
        return state_dict[key]

    state_num_classes: Optional[int] = None
    if "fusion_gate.0.weight" in state_dict:
        state_num_classes = state_dict["fusion_gate.0.weight"].shape[0]
    else:
        candidate_names = [
            "classifier.7.weight",
            "classifier.weight",
            "output_layer.weight",
            "model.classifier.weight",
            "classifier.7.bias",
            "classifier.bias",
            "output_layer.bias",
            "model.classifier.bias",
        ]
        for name in candidate_names:
            key = _prefixed(name)
            if key in state_dict:
                state_num_classes = state_dict[key].shape[0]
                break

    num_labels = int(metadata.get("num_labels", len(label_to_idx)))
    if state_num_classes is not None:
        if num_labels != state_num_classes:
            print(
                f"⚠️ Orion checkpoint output dimension ({state_num_classes}) "
                f"differs from metadata num_labels ({num_labels}); using state_dict value."
            )
        num_labels = state_num_classes
    if num_labels != len(label_to_idx):
        reconstructed = _reconstruct_labels_from_dataset(metadata, num_labels)
        if reconstructed is not None:
            label_to_idx = reconstructed
        else:
            raise RuntimeError(
                "Label mapping size does not match the classifier output dimension. "
                "Regenerate the checkpoint metadata so label_to_idx covers every class."
            )

    vocab: Optional[Dict[str, int]] = metadata.get("vocab")
    tokenizer = None
    tokenizer_cache = None
    embedding_fn = None
    encoder_label = base_encoder_type

    if base_encoder_type == "bilstm":
        if vocab is None:
            expected_vocab_size = _state_tensor("embedding.weight").shape[0]
            vocab = _reconstruct_vocab(metadata, expected_vocab_size)
            if vocab is None:
                raise ValueError("BiLSTM encoder requires a vocabulary in metadata.")
        embedding_dim = metadata.get("embedding_dim")
        hidden_dim = metadata.get("hidden_dim")
        ffn_dim = metadata.get("ffn_dim")
        num_layers = metadata.get("encoder_layers")
        attention_heads = metadata.get("attention_heads")
        dropout = metadata.get("dropout", 0.3)

        embedding_dim = int(embedding_dim or _state_tensor("embedding.weight").shape[1])
        hidden_dim = int(hidden_dim or _state_tensor("lstm.weight_hh_l0").shape[1])
        ffn_dim = int(ffn_dim or _state_tensor("ffn.0.weight").shape[0])
        num_layers = int(num_layers or _infer_lstm_layers(state_dict, prefix=base_prefix))
        embed_dim = hidden_dim * 2
        candidate_heads = [attention_heads, 4, 8, 2, 1]
        attention_heads_final: Optional[int] = None
        for candidate in candidate_heads:
            if candidate is None:
                continue
            candidate_int = int(candidate)
            if candidate_int > 0 and embed_dim % candidate_int == 0:
                attention_heads_final = candidate_int
                break
        if attention_heads_final is None:
            # Fall back to the largest divisor of embed_dim not exceeding 8
            for candidate_int in range(min(8, embed_dim), 0, -1):
                if embed_dim % candidate_int == 0:
                    attention_heads_final = candidate_int
                    break
        if attention_heads_final is None:
            attention_heads_final = 1

        conv_kernel_sizes_meta = metadata.get("bilstm_conv_kernels")
        conv_kernel_sizes_values: List[int] = []
        if isinstance(conv_kernel_sizes_meta, list):
            for value in conv_kernel_sizes_meta:
                try:
                    numeric = int(value)
                except (TypeError, ValueError):
                    continue
                if numeric > 0:
                    conv_kernel_sizes_values.append(numeric)
        elif isinstance(conv_kernel_sizes_meta, str):
            parts = [part.strip() for part in conv_kernel_sizes_meta.split(",") if part.strip()]
            for part in parts:
                try:
                    numeric = int(part)
                except ValueError:
                    continue
                if numeric > 0:
                    conv_kernel_sizes_values.append(numeric)
        conv_kernel_sizes = sorted(set(conv_kernel_sizes_values))
        state_kernels, state_channels = _infer_conv_head(state_dict, prefix=base_prefix)
        if not conv_kernel_sizes and state_kernels:
            conv_kernel_sizes = state_kernels
        conv_channels_meta = metadata.get("bilstm_conv_channels")
        conv_channels_final = int(conv_channels_meta) if conv_channels_meta else (state_channels or 256)
        conv_dropout = float(metadata.get("bilstm_conv_dropout", 0.2))
        use_conv_head = bool(conv_kernel_sizes)

        model = IntentClassifier(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_labels,
            dropout=float(dropout or 0.0),
            num_layers=num_layers,
            attention_heads=attention_heads_final,
            ffn_dim=ffn_dim,
            use_conv_head=use_conv_head,
            conv_kernel_sizes=conv_kernel_sizes if use_conv_head else None,
            conv_channels=conv_channels_final,
            conv_dropout=conv_dropout,
        )
    elif base_encoder_type == "transformer":
        transformer_model = metadata.get("transformer_model")
        if not transformer_model:
            raise ValueError("Transformer metadata missing model name.")
        base_model = TransformerIntentModel(transformer_model, num_classes=num_labels)
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'transformers' package is required for transformer inference."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        if hasattr(base_model, "model") and hasattr(base_model.model, "resize_token_embeddings"):
            base_model.model.resize_token_embeddings(len(tokenizer))
        model = base_model
    elif base_encoder_type == "st":
        base_model, embedding_fn = _load_sentence_transformer(metadata, num_labels)
        model = base_model
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported encoder type: {base_encoder_type}")

    if is_emotion_adaptive:
        emotion_dim = 0
        if "emotion_transform.0.weight" in state_dict:
            emotion_dim = state_dict["emotion_transform.0.weight"].shape[1]
        emotion_config = metadata.get("emotion_reasoner")
        fusion_dropout = 0.1
        if isinstance(emotion_config, dict):
            fusion_dropout = float(emotion_config.get("fusion_dropout", fusion_dropout))
        if emotion_dim <= 0:
            raise ValueError("EmotionallyAdaptiveModel checkpoint missing emotion transform weights.")
        model = EmotionallyAdaptiveModel(
            model,
            num_classes=num_labels,
            num_emotions=emotion_dim,
            dropout=fusion_dropout,
        )
        encoder_label = f"{base_encoder_type}+emotion"

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return OrionResources(
        model=model,
        device=device,
        label_to_idx=label_to_idx,
        vocab=vocab,
        max_seq_len=max_seq_len,
        encoder_type=encoder_label,
        tokenizer=tokenizer,
        tokenizer_cache=None,
        embedding_fn=embedding_fn,
        model_dir=model_dir,
    )


def format_top_predictions(predictions: Sequence[Tuple[str, float]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for idx, (label, score) in enumerate(predictions, start=1):
        rows.append([str(idx), label, f"{score * 100:.2f}%"])
    return rows


Message = Dict[str, str]


def classify_text(text: str, history: List[Message], resources: OrionResources):
    if resources.vocab is None and resources.encoder_type == "bilstm":
        raise RuntimeError("Vocabulary not loaded for BiLSTM encoder.")

    if not text or not text.strip():
        prompt = "⚠️ Please enter a message for Orion to analyse."
        return history, prompt, "", [], "", history

    prediction = predict_with_trace(
        resources.model,
        text,
        vocab=resources.vocab or {},
        label_to_idx=resources.label_to_idx,
        max_len=resources.max_seq_len,
        device=resources.device,
        tokenizer=resources.tokenizer,
        tokenizer_cache=resources.tokenizer_cache,
        embedding_model=resources.embedding_fn,
        top_k=5,
    )

    response = generate_response(prediction.label, text)

    summary_md = (
        f"### Prediction\n"
        f"- **Intent:** `{prediction.label}`\n"
        f"- **Confidence:** {prediction.confidence * 100:.2f}%\n"
        f"- **Model encoder:** {resources.encoder_type}\n"
    )
    if resources.model_dir is not None:
        summary_md += f"- **Model checkpoint:** `{resources.model_dir.name}`\n"

    if prediction.top_predictions:
        summary_md += "\n**Ranked intents:**\n"
        for label, score in prediction.top_predictions:
            summary_md += f"- {label}: {score * 100:.2f}%\n"

    response_md = (
        f"### Orion's Suggested Reply\n"
        f"- **Strategy:** {response.strategy}\n\n"
        f"{response.message}"
    )
    if response.basis:
        response_md += f"\n\n_Basis:_ {response.basis}"

    assistant_content = (
        f"{response.message}\n\nIntent: **{prediction.label}** "
        f"({prediction.confidence * 100:.1f}% confidence)"
    )

    updated_history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": assistant_content},
    ]

    table_rows = format_top_predictions(prediction.top_predictions)

    return updated_history, summary_md, response_md, table_rows, "", updated_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Orion intent demo UI.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to a specific Orion model directory (defaults to the newest).",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host/IP to bind the UI server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the UI server.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio sharing (tunnels the app for remote access).",
    )
    args = parser.parse_args()

    resources = build_orion_resources(args.model_dir)

    description = (
        "Leverage the promoted Orion intent classifier to analyse your text. "
        "Orion predicts the dominant intent, surfaces high-confidence alternatives, "
        "and suggests a contextual response strategy."
    )

    with gr.Blocks(theme=gr.themes.Soft(), css="""
    #orion-container {
        max-width: 840px;
        margin: auto;
    }
    #orion-output, #orion-response {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #312e81 100%);
        color: #f8fafc;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 18px 35px rgba(15, 23, 42, 0.35);
    }
    #orion-output strong, #orion-response strong {
        color: #fbbf24;
    }
    #orion-header {
        text-align: center;
        padding-bottom: 0.5rem;
    }
    .gradio-button.primary {
        background: linear-gradient(120deg, #f97316, #fb7185);
        border: none;
        color: white;
    }
    """) as demo:
        gr.Markdown("# Orion Intent Companion", elem_id="orion-header")
        gr.Markdown(description, elem_id="orion-container")

        history_state = gr.State([])

        with gr.Column(elem_id="orion-container"):
            with gr.Row():
                user_input = gr.Textbox(
                    label="Message",
                    placeholder="Type or paste a message for Orion…",
                    lines=3,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", variant="primary")

            chatbot = gr.Chatbot(label="Conversation", show_label=True, type="messages")

            summary = gr.Markdown(label="Intent Summary", elem_id="orion-output")
            response_panel = gr.Markdown(label="Suggested Reply", elem_id="orion-response")
            top_table = gr.Dataframe(
                headers=["Rank", "Intent", "Confidence"],
                datatype=["str", "str", "str"],
                row_count=(0, "dynamic"),
                label="Top candidate intents",
            )

        def _predict(user_text, history):
            return classify_text(user_text, history, resources)

        send_btn.click(
            _predict,
            inputs=[user_input, history_state],
            outputs=[chatbot, summary, response_panel, top_table, user_input, history_state],
        )
        user_input.submit(
            _predict,
            inputs=[user_input, history_state],
            outputs=[chatbot, summary, response_panel, top_table, user_input, history_state],
        )

    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=False)


if __name__ == "__main__":
    main()
