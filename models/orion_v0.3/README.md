# Orion V0.3 snapshot

- **Validation accuracy:** 69.38%
- **Training accuracy at best epoch:** 82.49%
- **Best epoch/stage:** 18.0 (supervised)
- **Labelled dataset:** 4344 examples across 49 intents
- **Pseudo-labelled additions:** 0
- **Encoder type:** transformer (prajjwal1/bert-tiny)
- **Effective learning rate:** 1.00e-04
- **Dataset checksum:** `ed17b6e609ae60e6efa59e347600a28415912dee`
- **Trainer version:** orion-trainer-0.3
- **Run timestamp (UTC):** 2025-10-22T12:22:09Z
- **Promoted to orion_v0.1:** yes

## Promotion rules
- Promote a run to `orion_v0.1` only when its validation accuracy exceeds the previous best by more than 0.0100 percentage points.
- Keep every training run under `models/runs/` for auditing and reproducibility.
- Update the metadata and metrics files alongside the weights when a promotion occurs.