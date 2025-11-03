# Orion V0.4 snapshot

- **Validation accuracy:** 1.94%
- **Training accuracy at best epoch:** 2.52%
- **Best epoch/stage:** 1.0 (supervised)
- **Labelled dataset:** 500 examples across 15 intents
- **Pseudo-labelled additions:** 0
- **Synthetic self-play additions:** 0
- **Encoder type:** st (all-MiniLM-L6-v2)
- **Effective learning rate:** 5.00e-04
- **Dataset checksum:** `7e50fa10a18ecc845f3f865fcb0bc31867995dfb`
- **Trainer version:** orion-trainer-0.7
- **Run timestamp (UTC):** 2025-11-03T15:08:30Z
- **Promoted to orion_v0.4:** no

## Promotion rules
- Promote a run to `orion_v0.4` only when its validation accuracy exceeds the previous best by more than 0.0100 percentage points.
- Keep every training run under `models/runs/` for auditing and reproducibility.
- Update the metadata and metrics files alongside the weights when a promotion occurs.