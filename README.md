# Solarus

Solarus is a compact intent-classification playground built around a curated
corpus of 834 short utterances spanning 36 conversational intents (greetings,
questions, reminders, announcements, jokes, weather updates, definitions, and
more). The training script couples modern optimisation utilities with
self-training so that the model can continue improving whenever fresh unlabeled
text is available.

## Dataset

- `data/intent_dataset.csv` – 834 labelled examples with balanced coverage over
  36 intents.
- `data/unlabeled_pool.csv` – an optional pool of unlabeled utterances that can
  be pseudo-labelled during self-training rounds.

The dataset intentionally prunes templated "Please consider"/"Always remember"
phrases and replaces them with human-written reminders, announcements, and
factual statements to provide cleaner supervision.

## Training script highlights

`train_intent_classifier.py` orchestrates the full workflow:

- stratified train/validation split with reproducible seeding
- choice of encoder:
  - `bilstm` – bidirectional LSTM with attention pooling and projection head
  - `transformer` – Hugging Face encoder fine-tuning (defaults to
    `prajjwal1/bert-tiny`)
  - `st` – frozen sentence-transformer encoder (`all-MiniLM-L6-v2` by default)
    with a lightweight feed-forward classifier; sentence embeddings are cached
    to avoid recomputation across epochs
- optimised training loop with AdamW, One-Cycle or cosine schedulers, label
  smoothing, gradient clipping, and optional mixed precision (AMP)
- configurable self-training rounds that pseudo-label high-confidence examples
  from an unlabeled pool and continue optimisation with down-weighted loss
- per-epoch metric logging, dataset statistics capture, and run metadata/metrics
  JSON export for auditability
- automatic run registry: every training run is archived under `models/runs/`
  and the best checkpoint is only promoted to `models/orion_v0.1/` when its
  validation accuracy surpasses the current best by more than the configured
  tolerance (default 0.01 percentage points)
- templated response generator that showcases predictions on sample utterances
  after training completes

## Requirements

- Python 3.12+
- [PyTorch](https://pytorch.org/) (CPU builds are sufficient)
- [sentence-transformers](https://www.sbert.net/)
- [transformers](https://huggingface.co/docs/transformers/index)

Install the dependencies with:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install sentence-transformers transformers
```

## Training examples

Fine-tune the sentence-transformer head (recommended for the best accuracy at
present):

```bash
python train_intent_classifier.py \
  --encoder-type st \
  --epochs 60 \
  --batch-size 48 \
  --learning-rate 3e-4 \
  --self-train-rounds 0
```

Run a compact transformer baseline:

```bash
python train_intent_classifier.py \
  --encoder-type transformer \
  --transformer-model prajjwal1/bert-tiny \
  --transformer-learning-rate 5e-5 \
  --epochs 10
```

Train the BiLSTM + attention encoder with self-training:

```bash
python train_intent_classifier.py \
  --encoder-type bilstm \
  --epochs 20 \
  --batch-size 32 \
  --self-train-rounds 2 \
  --self-train-epochs 2 \
  --unlabeled-dataset data/unlabeled_pool.csv
```

### Promotion guard rails

- Every run is stored under `models/runs/<timestamp>__accXXpYY__[tag]/` with the
  model weights, metadata, metrics, and an accuracy summary README.
- The high-watermark checkpoint lives in `models/orion_v0.1/`. Promotion occurs
  only when the new validation accuracy beats the prior best by more than the
  `--promotion-tolerance` threshold (default: 0.0001 absolute / 0.01%).
- Promotion metadata (dataset checksum, hyperparameters, timestamp, etc.) is
  persisted so you can audit how each Orion snapshot was produced.

## Outputs

After a training run, check:

- `models/runs/…/model.pt` – the trained weights for that run
- `models/runs/…/metadata.json` – dataset statistics, vocabulary or embedding
  settings, self-training history
- `models/runs/…/metrics.json` – best epoch metrics and promotion decision
- `models/orion_v0.1/` – the currently promoted best-performing checkpoint along
  with an auto-generated README summarising its accuracy and settings

These artefacts make it straightforward to compare encoder choices and retrain
only when the Orion benchmark improves.
