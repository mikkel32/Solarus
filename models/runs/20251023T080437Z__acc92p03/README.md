# Orion V0.4 snapshot

- **Validation accuracy:** 92.03%
- **Training accuracy at best epoch:** 98.50%
- **Best epoch/stage:** 19.0 (supervised)
- **Labelled dataset:** 33850 examples across 49 intents
- **Pseudo-labelled additions:** 0
- **Synthetic self-play additions:** 106
- **Encoder type:** transformer (prajjwal1/bert-tiny)
- **Effective learning rate:** 1.00e-04
- **Dataset checksum:** `94719fdeee95bc1287057ad06ffa3ae849418ceb`
- **Trainer version:** orion-trainer-0.4
- **Run timestamp (UTC):** 2025-10-23T08:04:37Z
- **Promoted to orion_v0.4:** yes

## Promotion rules
- Promote a run to `orion_v0.4` only when its validation accuracy exceeds the previous best by more than 0.0100 percentage points.
- Keep every training run under `models/runs/` for auditing and reproducibility.
- Update the metadata and metrics files alongside the weights when a promotion occurs.