# GAZELLE Training Architecture Overview

`gazelle-dinov3` applies a customized training pipeline for gaze target estimation. The key implementation points live in `scripts/train_gazefollow.py`, `scripts/train_vat.py`, `gazelle/model.py`, `gazelle/backbone.py`, and `gazelle/dataloader.py`. This note summarizes how those parts interact.

## High-Level Flow
- A DINOv2/DINOv3 Vision Transformer provides backbone features. Each sample consists of an image plus a bounding box for the person whose gaze direction we want to predict.
- For every person in the image, the model duplicates the feature map, injects a head-region token derived from the bounding box, and predicts a 64×64 heatmap that marks gaze probability. An optional in/out classifier reports whether the gaze target lies inside the frame.
- Training and evaluation rely on the GazeFollow and VideoAttentionTarget datasets with metrics such as AUC, average L2, and minimum L2 distances.

## Model Architecture (`gazelle/model.py`)

### Backbone Projection and Positional Encoding
- Backbone outputs are passed through a 1×1 convolution to the model's working dimension (default 256). A 2D sinusoidal positional embedding is added to maintain spatial information.

### Person-Aware Tokenization
- `utils.repeat_tensors` duplicates the feature map for each person instance.
- Head bounding boxes become 32×32 binary masks. Multiplying these masks with a learned `head_token` embedding injects person-specific cues into the features.

### Transformer Decoder Head
- A stack of `timm` ViT `Block`s (default three layers, 8 heads, MLP ratio 4) processes the flattened sequence.
- When in/out prediction is enabled, a special token is prepended. The `inout_head` MLP consumes that token to output a binary probability.

### Heatmap Reconstruction
- Transformer outputs are reshaped back to the spatial grid, passed through a ConvTranspose2d → Conv2d head, optionally activated by sigmoid, and resized to 64×64. `utils.split_tensors` restores the per-image grouping.
- `GazeLLE_ONNX` mirrors this design for export, embedding RGB conversion and normalization inside the graph.

## Backbone Handling (`gazelle/backbone.py`)
- `DinoV2Backbone` and `DinoV3Backbone` implement the shared `Backbone` interface, returning patch-level feature maps for a given input size.
- DINOv3 variants support `interaction_indexes`, trimming the transformer to specific blocks. A custom `VisionTransformer` adapter loads pre-trained checkpoints when necessary.
- `configure_backbone_finetune` first freezes the backbone and then selectively re-enables gradients on the last *N* transformer blocks plus normalization layers, positional embeddings, and CLS tokens.
- `get_backbone_num_blocks` exposes the number of transformer blocks available for progressive unfreezing.

## Data Preparation and Targets (`gazelle/dataloader.py`, `gazelle/utils.py`)
- The dataloader expands the dataset JSON into person-level samples.
- Training-time augmentations include:
  - Photometric distortion (`RandomPhotometricDistort`)
  - Random crops constrained to keep the head and (if in-frame) gaze target
  - Horizontal flips with matching coordinate updates
  - Bounding-box jitter
- Supervision uses Gaussian heatmaps (σ=3) on a 64×64 grid. Evaluation utilities compute dataset-specific AUC and L2 metrics.

## Training Loop Highlights

### Optimizer and Losses
- Adam optimizes both head and backbone parameters with separate parameter groups. `--lr` governs the head; `--backbone_lr` and optional weight decay apply to the unfrozen backbone subset.
- The heatmap loss is `BCELoss` by default. When `--disable_sigmoid` is set, the model emits logits and the code switches to `BCEWithLogitsLoss`.
- `BCEWithLogitsLoss` integrates the sigmoid transformation inside the loss function, which keeps gradients well-scaled for confident predictions and prevents numerical underflow that can arise from applying a separate sigmoid to large-magnitude logits; this stability is especially helpful when progressive unfreezing or distillation temporarily produces extreme activations.
- VideoAttentionTarget training adds a BCE loss for the in/out classifier, scaled by `--inout_loss_lambda`; evaluation reports average precision.

### Knowledge Distillation
- Setting `--distill_weight > 0` spawns a frozen teacher from `DEFAULT_TEACHER_CKPTS`.
- A cosine temperature schedule (`_cosine_anneal`) interpolates between `distill_temp_start` and `distill_temp_end` across training steps.
- Student and teacher heatmaps are flattened, converted to probability distributions, and compared via KL divergence times the square of the temperature. When logits are emitted, they are used directly; otherwise the code applies `torch.logit`.

### Progressive Unfreeze
- With `--finetune`, the script unfreezes the last `--finetune_layers` backbone blocks at the start.
- After `--initial_freeze_epochs`, every `--unfreeze_interval` epochs one additional block becomes trainable until all blocks are included. Logging announces the schedule. `--disable_progressive_unfreeze` locks the number of trainable blocks.
- Progressive unfreezing preserves the stability of a strong frozen backbone during the early, noise-prone training phase; once the head converges, gradually exposing deeper layers lets the optimizer adapt features without catastrophic forgetting, leading to smoother loss curves and stronger final accuracy.

### Stability and Throughput
- Mixed precision (`torch.amp.autocast` + `GradScaler`) activates via `--use_amp`. `--grad_clip_norm` applies gradient clipping after optional unscaling.
- Training logs per `log_iter` iterations to TensorBoard, including distillation loss and temperature when enabled. Validation uses `tqdm` progress bars.

### Evaluation and Checkpoints
- A cosine annealing LR scheduler (`CosineAnnealingLR`) spans the full run with `T_max=max_epochs` and `eta_min=1e-7`.
- Each epoch saves a checkpoint containing model weights (optionally excluding the backbone), optimizer, scheduler, AMP scaler, RNG state, and run arguments. Old checkpoints are pruned with `prune_epoch_checkpoints` / `prune_best_checkpoints`.
- The best model is tracked using minimum L2. When resuming, saved arguments restore hyperparameters and TensorBoard counters (`purge_step`) stay consistent.

## Example Training Command
```bash
uv run python scripts/train_gazefollow.py \
--data_path data/gazefollow_extended \
--model_name gazelle_dinov3_vit_tiny \
--exp_name gazelle_dinov3_s_ft_bcelogits_prog \
--max_epochs 45 --batch_size 64 --lr 1e-3 \
--finetune --finetune_layers 2 \
--initial_freeze_epochs 10 --unfreeze_interval 3 \
--grad_clip_norm 1.0 --use_amp --disable_sigmoid
```
This mirrors the configuration from `README.md`, combining progressive backbone unfreezing with logit-based heatmap training. To enable distillation, add `--distill_teacher` and `--distill_weight`.

---

By coupling high-resolution ViT features with person-aware tokens, knowledge distillation, and staged backbone fine-tuning, GAZELLE balances data efficiency and training stability for gaze target localization.
