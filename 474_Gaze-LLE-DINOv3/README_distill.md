# GazeLLE Distillation Guide

This note summarizes how to train a smaller GazeLLE variant (e.g. `gazelle_dinov3_vit_tiny`) under a teacher such as `gazelle_dinov3_vits16plus` using the distillation hooks wired into the training scripts.

## 1. Overview
- Distillation is disabled by default. It becomes active only when `--distill_weight` is set to a positive value.
- Both `scripts/train_gazefollow.py` and `scripts/train_vat.py` now expose a common set of knobs:
  - `--distill_teacher`: teacher model name (defaults to `gazelle_dinov3_vits16plus`).
  - `--distill_weight`: scalar weight applied to the auxiliary KL loss.
  - `--distill_temp_start`: temperature used at the beginning of training (default `1.0`).
  - `--distill_temp_end`: temperature reached at the last training step via cosine annealing (default `4.0`).
  - `--distill_teacher_ckpt`: optional override pointing to the teacher’s checkpoint (defaults to a lookup in `./ckpts/`).
- When enabled, the teacher model is loaded in evaluation mode, frozen, and kept in `torch.no_grad()` contexts during training.

## 2. Enabling Distillation
```bash
# Example: distilling the ViT-Tiny student on GazeFollow
uv run python scripts/train_gazefollow.py \
--model_name gazelle_dinov3_vit_tiny \
--distill_teacher gazelle_dinov3_vits16plus \
--distill_weight 0.3 \
--distill_temp_end 4.0
```

```bash
# Example: distilling the in/out model on VAT
uv run python scripts/train_vat.py \
--model gazelle_dinov3_vit_tinyplus \
--distill_teacher gazelle_dinov3_vits16plus \
--distill_weight 0.3 \
--distill_temp_end 4.0
```

Passing `--distill_weight 0` (or omitting the flag) keeps the previous training behaviour.

## 3. Teacher Choice
- Start with `gazelle_dinov3_vits16plus`: it offers a strong signal while remaining light enough to co-train with the student.
- Once the pipeline is stable, consider swapping to `gazelle_dinov3_vitb16` for potential accuracy gains. Expect to revisit loss weights because the representational gap grows with the larger teacher.
- If the student and teacher share the same architecture, expect marginal benefit; the scripts still allow this configuration but emit a warning.

The training scripts try to locate pretrained heads for the most common teachers automatically:

- `gazelle_dinov3_vit_tiny` → `./ckpts/gazelle_dinov3_vit_tiny.pt`
- `gazelle_dinov3_vit_tinyplus` → `./ckpts/gazelle_dinov3_vit_tinyplus.pt`
- `gazelle_dinov3_vits16` → `./ckpts/gazelle_dinov3_vits16.pt`
- `gazelle_dinov3_vits16plus` → `./ckpts/gazelle_dinov3_vits16plus.pt`
- `gazelle_dinov3_vitb16` → `./ckpts/gazelle_dinov3_vitb16.pt`
- `gazelle_dinov2_vitb14` → `./ckpts/gazelle_dinov2_vitb14.pt`
- `gazelle_dinov2_vitl14` → `./ckpts/gazelle_dinov2_vitl14.pt`
- `gazelle_dinov2_vitb14_inout` → `./ckpts/gazelle_dinov2_vitb14_inout.pt`
- `gazelle_dinov2_vitl14_inout` → `./ckpts/gazelle_dinov2_vitl14_inout.pt`

If a teacher name is missing or you keep checkpoints elsewhere, pass `--distill_teacher_ckpt /path/to/file.pt`.

## 4. Loss Formulation
- Student heatmaps still optimise BCE/BCEWithLogits exactly as before.
- The auxiliary term now uses **temperature-scaled KL divergence**:
  1. Student/teacher heatmaps are converted to logits (`torch.logit` when they arrive as probabilities).
  2. Logits are divided by the scheduled temperature.
  3. `torch.log_softmax` + `torch.softmax` produce log-probabilities/probabilities over the flattened heatmap.
  4. `nn.KLDivLoss(reduction='batchmean')` is multiplied by `temperature²` (standard KD normalisation).
- The total loss becomes `L_total = L_supervised + distill_weight * KL_T`.

## 5. Temperature Schedule
- Cosine annealing gradually morphs the temperature from `distill_temp_start` to `distill_temp_end` across all training steps:
  `T(s) = T_start + 0.5 * (1 - cos(pi * s)) * (T_end - T_start)`, where `s` is the normalised step counter.
- Higher temperatures soften the teacher distribution, allowing the student to focus on relative preferences instead of exact peaks. Starting low keeps early training close to the supervised baseline, while ending high emphasises the teacher near convergence.
- The defaults (`1.0 → 4.0`) work well as a first attempt. If the teacher is noisy, keep the end temperature smaller; if the student underfits the teacher, raise it.

## 6. Picking `distill_weight`
- Begin around **0.3** — the KL term stays comparable to the BCE loss during early training.
- Run a short sweep (`0.1`, `0.3`, `0.5`, `1.0`) and watch:
  - If BCE stagnates or spikes, the weight is too high.
  - If the KL loss refuses to drop, the weight is too small.
- Optionally ramp `distill_weight` from `0` during the first epoch if you still observe instability.

## 7. Logging & Monitoring
- New TensorBoard tags:
  - `train/distill_loss` reports the KL term.
  - `train/distill_temperature` tracks the cosine schedule.
- Keep tracking the existing metrics (AUC, L2, AP). A healthy run will show the distill loss trending down while the supervised metrics continue improving.

## 8. Checkpoint & Resume Behaviour
- The new CLI flags are stored inside checkpoints. Resuming a run automatically restores the teacher, weight, and temperature parameters and prints a warning if the CLI input differs.
- Training checkpoints now persist **all** learnable weights, including the backbone when it is being fine-tuned. Older checkpoints created before this change may still lack backbone tensors; loading them triggers a warning but remains supported.
- When you wish to reuse a checkpoint as the distillation teacher, you can pass it via `--distill_teacher_ckpt` and the loader will hydrate both backbone and head. By default the scripts still fall back to the curated files in `./ckpts/` (e.g. `gazelle_dinov3_vits16plus.pt`, `gazelle_dinov3_vitb16.pt`).

## 9. Practical Tips
- **GPU memory**: running both student and teacher forward passes roughly doubles memory usage. Use AMP (`--use_amp`) or smaller batch sizes if you encounter OOM errors.
- **Data augmentations**: strong augmentations help the student generalize when imitating a larger teacher. Ensure teacher and student receive the *same* inputs to keep the supervision consistent.
- **Sanity checks**:
  - Set `--distill_weight` to a very large value (e.g. `10`) for a few iterations; the student heatmaps should quickly mimic the teacher. Revert afterwards.
  - Run one training step with `--distill_weight 0` and confirm the loss matches the historical baseline.
- **Teacher checkpoints**: verify the teacher’s base performance before distilling. Garbage in leads to garbage out. Ensure the checkpoint you point to contains backbone weights (all checkpoints saved by the current training scripts do); otherwise the loader will warn and continue with the available parameters.

## 10. Next Steps
1. Launch a short training run with `--distill_weight 0.3` to confirm the pipeline works end-to-end.
2. Evaluate the student against the previous non-distilled checkpoint to quantify gains.
3. Explore mid-level feature or attention-map matching if the student accuracy plateaus.

With these pieces in place, you can iterate on teacher choices, loss weighting, and additional hints to tailor the distillation strategy to your needs.
