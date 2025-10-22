# Debug Model snapshot

- **Validation accuracy:** 20.87%
- **Training accuracy at best epoch:** 9.56%
- **Best epoch/stage:** 1.0 (supervised)
- **Labelled dataset:** 4344 examples across 49 intents
- **Pseudo-labelled additions:** 0
- **Encoder type:** bilstm
- **Effective learning rate:** 3.00e-04
- **Dataset checksum:** `ed17b6e609ae60e6efa59e347600a28415912dee`
- **Trainer version:** orion-trainer-0.4
- **Run timestamp (UTC):** 2025-10-22T14:41:35Z
- **Promoted to debug_model:** no

## Promotion rules
- Promote a run to `debug_model` only when its validation accuracy exceeds the previous best by more than 0.0100 percentage points.
- Keep every training run under `models/runs/` for auditing and reproducibility.
- Update the metadata and metrics files alongside the weights when a promotion occurs.