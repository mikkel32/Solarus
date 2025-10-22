# Orion V0.4 snapshot

- **Validation accuracy:** 0.00%
- **Training accuracy at best epoch:** 0.00%
- **Best epoch/stage:** 1.0 (-)
- **Labelled dataset:** 4344 examples across 49 intents
- **Pseudo-labelled additions:** 0
- **Encoder type:** bilstm
- **Effective learning rate:** 0.00e+00
- **Dataset checksum:** `ed17b6e609ae60e6efa59e347600a28415912dee`
- **Trainer version:** orion-trainer-0.4
- **Run timestamp (UTC):** 2025-10-22T13:45:32Z
- **Promoted to orion_v0.4:** no

## Promotion rules
- Promote a run to `orion_v0.4` only when its validation accuracy exceeds the previous best by more than 0.0100 percentage points.
- Keep every training run under `models/runs/` for auditing and reproducibility.
- Update the metadata and metrics files alongside the weights when a promotion occurs.