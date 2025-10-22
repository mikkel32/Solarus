# Orion V0.1 snapshot

- **Validation accuracy:** 34.03%
- **Training accuracy at best epoch:** 87.77%
- **Best epoch/stage:** 42.0 (self-train-1)
- **Labelled dataset:** 747 examples across 36 intents
- **Pseudo-labelled additions:** 3
- **Dataset checksum:** `a86bd73c7203a5d114cfd5fbb39f0976abf1c0bd`
- **Trainer version:** orion-trainer-0.3
- **Run timestamp (UTC):** 2025-10-22T06:33:20Z
- **Promoted to orion_v0.1:** yes

## Promotion rules
- Promote a run to `orion_v0.1` only when its validation accuracy exceeds the previous best by more than 0.0500 percentage points.
- Keep every training run under `models/runs/` for auditing and reproducibility.
- Update the metadata and metrics files alongside the weights when a promotion occurs.