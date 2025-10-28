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

- stratified train/validation split with reproducible seeding, or
  stratified $k$-fold cross-validation via `--folds`
- choice of encoder:
  - `bilstm` – bidirectional LSTM with attention pooling and projection head
  - `transformer` – Hugging Face encoder fine-tuning (defaults to
    `prajjwal1/bert-tiny`)
  - `st` – frozen sentence-transformer encoder (`all-MiniLM-L6-v2` by default)
    with a lightweight feed-forward classifier; sentence embeddings are cached
    to avoid recomputation across epochs
- optimised training loop with AdamW, One-Cycle or cosine schedulers, label
  smoothing, gradient clipping, and optional mixed precision (AMP)
- gradient accumulation (`--grad-accumulation-steps`), exponential moving
  averages (`--ema-decay`, `--ema-use-for-eval`), stochastic weight averaging
  (`--swa-start-epoch`, `--swa-lr`, `--swa-anneal-epochs`), and token-level
  augmentation controls (`--augment-*`) to squeeze extra stability and
  performance out of each run
- optional ensemble-based knowledge distillation that averages the strongest
  cross-validation folds, filters confident teacher logits, and fine-tunes the
  promoted model with blended soft/hard targets via the `--distill-*` controls
- configurable self-training rounds that pseudo-label high-confidence examples
  from an unlabeled pool, with optional confidence-threshold decay
  (`--self-train-threshold-decay`) and confidence-weighted loss scaling
  (`--self-train-confidence-power`, `--self-train-max-weight-multiplier`)
- optional full-dataset consolidation via `--final-train-epochs`, allowing the
  promoted checkpoint to be fine-tuned on the entire labelled corpus (plus
  pseudo-labelled examples) before archiving a "final_full" artefact; the
  consolidation stage can retain the distillation loss with
  `--distill-keep-during-final` for additional regularisation
- richer metadata capture: per-epoch histories, class-wise precision/recall/F1,
  optimiser step counts, and augmentation statistics accompany every run in the
  `models/runs/` registry
- per-epoch metric logging, dataset statistics capture, and run metadata/metrics
  JSON export for auditability
- automatic aggregation of cross-validation metrics (mean/stdev) with promotion
  decisions based on the best-performing fold
- automatic run registry: every training run is archived under `models/runs/`
  and the best checkpoint is only promoted to `models/orion_v0.4/` when its
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

Run five-fold cross-validation with confidence-weighted pseudo-labelling:

```bash
python train_intent_classifier.py \
  --encoder-type bilstm \
  --folds 5 \
  --epochs 15 \
  --self-train-rounds 2 \
  --self-train-threshold-decay 0.05 \
  --self-train-min-threshold 0.7 \
  --self-train-confidence-power 1.5 \
  --self-train-max-weight-multiplier 3
```

Stack the advanced optimisation gadgets together (gradient accumulation, EMA,
SWA, and on-the-fly augmentation) while consolidating the final model on the
full dataset:

```bash
python train_intent_classifier.py \
  --encoder-type transformer \
  --folds 5 \
  --epochs 12 \
  --grad-accumulation-steps 4 \
  --ema-decay 0.995 --ema-start-epoch 2 --ema-use-for-eval \
  --swa-start-epoch 8 --swa-lr 2e-4 \
  --augment-probability 0.4 --augment-max-copies 2 --augment-max-transforms 3 \
  --final-train-epochs 5 --final-use-pseudo \
  --distill-epochs 3 --distill-alpha 0.6 --distill-temperature 2.5 \
  --distill-min-confidence 0.55 --distill-confidence-power 1.25 \
  --distill-max-weight-multiplier 4 --distill-max-teachers 3 \
  --distill-keep-during-final
```

Run a teacher-ensemble consolidation without additional augmentation:

```bash
python train_intent_classifier.py \
  --encoder-type transformer \
  --folds 5 \
  --epochs 10 \
  --final-train-epochs 4 \
  --distill-epochs 4 --distill-alpha 0.7 --distill-temperature 3.0 \
  --distill-min-confidence 0.6 --distill-max-teachers 2
```

### Promotion guard rails

- Every run is stored under `models/runs/<timestamp>__accXXpYY__[tag]/` with the
  model weights, metadata, metrics, and an accuracy summary README.
- The high-watermark checkpoint lives in `models/orion_v0.4/`. Promotion occurs
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
- `models/orion_v0.4/` – the currently promoted best-performing checkpoint along
  with an auto-generated README summarising its accuracy and settings

These artefacts make it straightforward to compare encoder choices and retrain
only when the Orion benchmark improves.

## Interactive demo UI

Spin up a conversational interface around the latest promoted Orion checkpoint
with:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install gradio
python scripts/orion_chat_ui.py
```

The app automatically selects the most recent `models/orion_v*/` directory,
loads its metadata, and serves a Gradio UI at http://localhost:7860. Enter any
utterance, press **Send**, and Orion will display the predicted intent, rank
its top alternatives, and propose a contextual reply strategy.

## Google Colab quickstart

Solarus now autodetects Google Colab runtimes. When the trainer runs inside
Colab it stages the labelled/unlabelled CSV files into a reusable workspace
under `/content/solarus_workspace`, and redirects run artefacts to a persistent
location (preferring Google Drive when it is mounted).

For a turnkey setup inside a Colab notebook:

1. Clone the repository or install the package in the runtime.
2. Execute `python scripts/colab_setup.py --install-extras all --install-pyright`.
   The helper script stages datasets in the workspace, creates a models
   directory, and installs optional dependencies.
3. Launch training as usual. Artefacts will be stored under
   `/content/solarus_models` by default, or in `MyDrive/Solarus/models` if
   Google Drive is mounted. Override the target with `--model-dir` or by
   setting `SOLARUS_COLAB_WORKSPACE`.

Static type checking inside Colab can be run with Pyright using the
`pyrightconfig.colab.json` profile, which points at Colab's library paths and
workspace mirrors. Local development now ships with a sibling
`pyrightconfig.json` tuned for maximum strictness—every diagnostic exposed by
Pyright is promoted to an error, `# type: ignore` escape hatches are disabled,
and reachability analysis plus experimental checkers are enabled. The
configuration narrows the include set to the repository's Python entry points
so that full-project scans remain fast while still flagging undefined names
("is not defined"), annotation gaps, redundant imports, and other issues long
before runtime.
