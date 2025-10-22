# Orion V0.1 snapshot

- **Validation accuracy:** 65.85%
- **Training accuracy at best epoch:** 76.72%
- **Best epoch/stage:** 43.0 (supervised)
- **Labelled dataset:** 834 examples across 36 intents
- **Pseudo-labelled additions:** 0
- **Encoder type:** st (all-MiniLM-L6-v2)
- **Effective learning rate:** 3.00e-04
- **Dataset checksum:** `2fa316085eff2e8ca7b296cc5684bee88bba5ced`
- **Trainer version:** orion-trainer-0.3
- **Run timestamp (UTC):** 2025-10-22T06:59:07Z
- **Promoted to orion_v0.1:** yes

## Promotion rules
- Promote a run to `orion_v0.1` only when its validation accuracy exceeds the previous best by more than 0.0500 percentage points.
- Keep every training run under `models/runs/` for auditing and reproducibility.
- Update the metadata and metrics files alongside the weights when a promotion occurs.

