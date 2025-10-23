"""Interactive UI for chatting with the latest Orion intent model."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import gradio as gr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_intent_classifier import (
    IntentClassifier,
    TransformerIntentModel,
    SentenceTransformerClassifier,
    generate_response,
    predict_with_trace,
)


@dataclass(frozen=True)
class OrionResources:
    model: torch.nn.Module
    device: torch.device
    label_to_idx: Dict[str, int]
    vocab: Optional[Dict[str, int]]
    max_seq_len: int
    encoder_type: str
    tokenizer: Optional[object] = None
    tokenizer_cache: Optional[object] = None
    embedding_fn: Optional[object] = None
    model_dir: Path | None = None


ORION_ROOT = REPO_ROOT / "models"


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


def _infer_lstm_layers(state_dict: Dict[str, torch.Tensor]) -> int:
    layers: set[int] = set()
    for key in state_dict:
        if key.startswith("lstm.weight_ih_l") and "reverse" not in key:
            layer_idx = int(key.split("_l")[1].split(".")[0])
            layers.add(layer_idx)
    return max(layers) + 1 if layers else 1


def _load_sentence_transformer(metadata: Dict[str, object], num_labels: int) -> Tuple[SentenceTransformerClassifier, Optional[object]]:
    model_name = metadata.get("sentence_transformer_model")
    hidden_dim = int(metadata.get("sentence_transformer_hidden_dim", 512))
    dropout = float(metadata.get("sentence_transformer_dropout", 0.2))
    if model_name is None:
        raise ValueError("Sentence-transformer metadata missing the model name.")
    try:
        from sentence_transformers import SentenceTransformer
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

    label_to_idx: Dict[str, int] = metadata["label_to_idx"]
    encoder_type = metadata.get("encoder_type", "bilstm")
    max_seq_len = int(metadata.get("max_seq_len", 128))
    num_labels = int(metadata.get("num_labels", len(label_to_idx)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(weights_path, map_location=device)

    vocab: Optional[Dict[str, int]] = metadata.get("vocab")
    tokenizer = None
    tokenizer_cache = None
    embedding_fn = None

    if encoder_type == "bilstm":
        if vocab is None:
            raise ValueError("BiLSTM encoder requires a vocabulary in metadata.")
        embedding_dim = metadata.get("embedding_dim")
        hidden_dim = metadata.get("hidden_dim")
        ffn_dim = metadata.get("ffn_dim")
        num_layers = metadata.get("encoder_layers")
        attention_heads = metadata.get("attention_heads")
        dropout = metadata.get("dropout", 0.3)

        embedding_dim = int(embedding_dim or state_dict["embedding.weight"].shape[1])
        hidden_dim = int(hidden_dim or state_dict["lstm.weight_hh_l0"].shape[1])
        ffn_dim = int(ffn_dim or state_dict["ffn.0.weight"].shape[0])
        num_layers = int(num_layers or _infer_lstm_layers(state_dict))
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

        model = IntentClassifier(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_labels,
            dropout=float(dropout or 0.0),
            num_layers=num_layers,
            attention_heads=attention_heads_final,
            ffn_dim=ffn_dim,
        )
    elif encoder_type == "transformer":
        transformer_model = metadata.get("transformer_model")
        if not transformer_model:
            raise ValueError("Transformer metadata missing model name.")
        model = TransformerIntentModel(transformer_model, num_classes=num_labels)
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'transformers' package is required for transformer inference."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    elif encoder_type == "st":
        model, embedding_fn = _load_sentence_transformer(metadata, num_labels)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return OrionResources(
        model=model,
        device=device,
        label_to_idx=label_to_idx,
        vocab=vocab,
        max_seq_len=max_seq_len,
        encoder_type=encoder_type,
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
