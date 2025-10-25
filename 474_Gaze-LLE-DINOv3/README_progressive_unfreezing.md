# Progressive Backbone Unfreezing in GazeLLE

This document explains how the training scripts manage staged unfreezing of ViT backbones, and how to tune the relevant knobs when finetuning on GazeFollow or VideoAttentionTarget.

## 1. Why Progressive Unfreezing?
- Large DINO backbones are pretrained on massive corpora; blindly finetuning all transformer blocks can cause catastrophic forgetting or unstable gradients.
- Gradually increasing the number of trainable layers keeps early updates focused on the task head, letting downstream loss signals stabilize before reaching deeper layers.
- The staging is especially helpful when moving from large teachers to compact students or when labelled data is scarce.

## 2. Key CLI Flags (`scripts/train_gazefollow.py`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--finetune` | off | Enables backbone finetuning logic. Without this flag the backbone stays frozen. |
| `--finetune_layers` | `2` | Number of transformer blocks initially unfrozen (counting from the last block backwards). Set `<= 0` to unfreeze the entire backbone immediately. |
| `--initial_freeze_epochs` | `10` | How many epochs to hold the initial `finetune_layers` before expanding the trainable set. |
| `--unfreeze_interval` | `3` | After the initial hold period, unfreeze one additional block every N epochs. |
| `--disable_progressive_unfreeze` | off | Keeps the trainable set fixed at `finetune_layers` for the entire run. |

The configuration is mirrored when resuming from checkpoints; the script restores these values automatically.

## 3. How It Works Internally
1. At startup the script determines the total number of backbone blocks (`get_backbone_num_blocks`).
2. The initial target is computed via `_target_layers_for_epoch(start_epoch)`. If `finetune_layers <= 0`, all blocks are unfrozen upfront.
3. During training, at the beginning of each epoch the script checks whether the schedule calls for more layers. When it does, `configure_backbone_finetune` is re-run to adjust `requires_grad` on the backbone modules and the optimizer parameter group is updated in place.
4. Logging: when progressive unfreezing is active the script prints a message like
   ```
   Progressive unfreeze scheduled after 10 epochs, adding one block every 3 epochs.
   ```
   and reports the final epoch expected to have all blocks unfrozen. If `final_unfreeze_epoch >= max_epochs`, a warning is emitted so you can expand training or adjust the schedule.

## 4. Practical Recipes

### Strong Head, Frozen Backbone (default)
```
--finetune          (omit this flag)
```
Use when the head alone is sufficient or when rapidly prototyping without expensive GPU time.

### Gentle Finetune
```
--finetune \
--finetune_layers 2 \
--initial_freeze_epochs 5 \
--unfreeze_interval 2
```
Only the last couple of transformer blocks learn at first; the remainder join gradually. Works well when you only need subtle adaptation.

### Full Backbone Adaptation
```
--finetune \
--finetune_layers 0 \
--disable_progressive_unfreeze
```
Unfreezes everything immediately. Suitable when you have ample labelled data and want the model to deviate more from the DINO prior.

### Slow Burn
```
--finetune \
--finetune_layers 4 \
--initial_freeze_epochs 15 \
--unfreeze_interval 5
```
Useful for very noisy datasets; it keeps deeper layers untouched until the head stabilizes.

## 5. Monitoring Tips
- Track the training loss around each scheduled unfreeze step. Sudden spikes may indicate too aggressive a schedule or the need for a smaller learning rate on the backbone parameter group.
- Inspect TensorBoard’s “grad_norm” (if you log it) or check diagnostics after checkpoints to ensure gradients aren’t exploding when new layers open up.
- Consider lowering `--backbone_lr` for the newly unfrozen layers (e.g. 10× smaller than the head LR) if you observe instability.

## 6. Interaction with Distillation
- Progressive unfreezing pairs well with the knowledge distillation pipeline. If the teacher already encodes strong priors, start with a partially frozen backbone to let the student head imitate the teacher first, then gradually expose deeper layers.
- When using large distillation weights, keep the unfreeze interval longer so the student can align with the teacher before adding more capacity.

## 7. Resume Behaviour
- The checkpoint stores the values of `finetune`, `finetune_layers`, `initial_freeze_epochs`, `unfreeze_interval`, and `disable_progressive_unfreeze`.
- When resuming, the script restores optimizer state, ensures the backbone layers match the saved schedule, and prints a warning if the CLI arguments differ from what the checkpoint encoded.

## 8. Troubleshooting
- **Nothing is being unfrozen**: ensure `--finetune` is passed and `--finetune_layers` is > 0. Also check that `--unfreeze_interval` isn’t `0` and that `max_epochs` exceeds `initial_freeze_epochs`.
- **OOM after unfreezing**: large backbones may exceed memory once gradients flow through deeper layers. Reduce batch size, enable AMP (`--use_amp`), or slow the unfreeze cadence.
- **Training regresses after full unfreeze**: consider capping the number of blocks (lower `finetune_layers`) or re-enabling progressive unfreezing with a longer schedule.

By tuning these parameters you can balance stability and adaptability, letting GazeLLE benefit from pretrained knowledge while still fitting your downstream gaze dataset.
