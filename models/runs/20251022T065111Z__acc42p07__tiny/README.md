# Orion V0.1 snapshot

- **Validation accuracy:** 42.07%
- **Training accuracy at best epoch:** 44.03%
- **Best epoch/stage:** 12.0 (supervised)
- **Labelled dataset:** 834 examples across 36 intents
- **Pseudo-labelled additions:** 0
- **Encoder type:** transformer (prajjwal1/bert-tiny)
- **Effective learning rate:** 1.00e-04
- **Dataset checksum:** `2fa316085eff2e8ca7b296cc5684bee88bba5ced`
- **Trainer version:** orion-trainer-0.3
- **Run timestamp (UTC):** 2025-10-22T06:51:11Z
- **Promoted to orion_v0.1:** yes

## Promotion rules
- Promote a run to `orion_v0.1` only when its validation accuracy exceeds the previous best by more than 0.0500 percentage points.
- Keep every training run under `models/runs/` for auditing and reproducibility.
- Update the metadata and metrics files alongside the weights when a promotion occurs.