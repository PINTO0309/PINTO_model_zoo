# 503_CDNet

### Paper
[Combined Depth Space based Architecture Search for Person Re-identification](https://arxiv.org/abs/2104.04163)

<img width="853" height="571" alt="image" src="https://github.com/user-attachments/assets/1e3f3171-1c70-4bc5-bad2-64e2d39a0243" />

### Models

- results on ReID tasks

  | model             | Market(mAP/rank-1) | Duke(mAP/rank-1) | MSMT17(mAP/rank-1) |
  | ----------------- | :----------------: | :--------------: | :----------------: |
  | cnet(scratch)     |     83.5/93.6      |    73.2/86.0     |     47.7/73.3      |
  | cdnet(scratch)    |     83.7/93.7      |    73.9/86.7     |     48.5/73.7      |
  | cdnet(pretrained) |     86.0/95.1      |    76.8/88.6     |     54.7/78.9      |

  - results on classification

  | model          | Cifar-100(acc/param) | ImageNet(acc/param) |
  | -------------- | -------------------- | ------------------- |
  | cdnet(scratch) | 82.1/2.3M            | 75.1/2.5M           |

## Original

https://github.com/solicucu/ReID

## ONNX custom

https://github.com/PINTO0309/ReID

---

# Fused CDNet ReID ONNX — Input / Output Specification

Applies to the three RoiAlign-fused CDNet models:

| Model | Internal crop size | Size |
|---|---|---:|
| `models/cdnet_top2_msmt17_prep_roialign_Nx3x256x128_opset17.onnx` | 256×128 | 7.5 MB |
| `models/cdnet_top2_msmt17_prep_roialign_Nx3x192x96_opset17.onnx` | 192×96 | 7.4 MB |
| `models/cdnet_top2_msmt17_prep_roialign_Nx3x128x64_opset17.onnx` | 128×64 | 7.3 MB |

## Input 1: `prep_frame_bgr` — float32 `[1, 3, H, W]`

The **raw BGR frame exactly as OpenCV decodes it** — values 0–255, no
scaling, no mean/std, no channel swap — transposed HWC→CHW with a batch dimension added:

```python
frame = np.ascontiguousarray(img_bgr.transpose(2, 0, 1)[None].astype(np.float32))
# e.g. [1, 3, 1080, 1920], value range 0.0–255.0
```

`H` and `W` are dynamic. All preprocessing lives **inside the graph** (see "What the graph does internally" below); preprocessing the frame yourself would apply it twice.

## Input 2: `prep_rois` — float32 `[N, 5]`

One row per person box:

```
[batch_idx, x1/W, y1/H, x2/W, y2/H]
```

- `batch_idx` is always `0.0` (the frame batch is 1).
- Coordinates are clipped to the frame, then **normalized to [0, 1]** by the frame width/height. Passing pixel coordinates is a bug.
- For a single TensorRT engine shape (no per-batch re-tuning), rows are zero-padded (`[0,0,0,0,0]`) up to a multiple of `batch_max` (default 64); only the first `N_real` output rows are consumed.

**Worked example** — a 1920×1080 frame, person box
(x1, y1, x2, y2) = (100, 50, 300, 450) px:

```
[0.0, 100/1920, 50/1080, 300/1920, 450/1080]
= [0.0, 0.052083, 0.046296, 0.156250, 0.416667]
```

## Output: `embeddings` — float32 `[N, 768]` (unnormalized)

L2-normalize before using cosine similarity:

```python
emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
```

## Minimal standalone example

```python
import cv2, numpy as np, onnxruntime as ort

sess = ort.InferenceSession(
    "models/cdnet_top2_msmt17_prep_roialign_Nx3x256x128_opset17.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

img = cv2.imread("frame.jpg")                       # BGR uint8
H, W = img.shape[:2]
frame = np.ascontiguousarray(img.transpose(2, 0, 1)[None].astype(np.float32))

boxes_px = np.array([[100, 50, 300, 450], [600, 80, 750, 500]], np.float64)
rois = np.array([[0.0, b[0]/W, b[1]/H, b[2]/W, b[3]/H] for b in boxes_px],
                np.float32)                          # [N, 5]

emb = sess.run(None, {"prep_frame_bgr": frame, "prep_rois": rois})[0]
emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
```

## What the graph does internally

```
prep_frame_bgr [1,3,H,W] ─┐
prep_rois      [N,5]     ─┴→ DynamicRoIAlign (GridSample, aligned=True)
                              → crop + resize to the model's crop size (BGR, 0–255)
                            → prep_/Slice   (steps=[-1] on axis 1: BGR → RGB)
                            → prep_/Sub     (mean·255 = [123.675, 116.28, 103.53], RGB order)
                            → prep_/Div     (std·255  = [58.395, 57.12, 57.375])
                            → CDNet backbone → embeddings [N,768]
```

The /255 and ImageNet normalization are folded into a single `(x − mean·255) / (std·255)` Sub+Div pair, matching the original CDNet training pipeline (solicucu/ReID: PIL RGB → ToTensor → Normalize with ImageNet mean/std @ 256×128).

## Do / Don't

- **Do** feed the OpenCV BGR frame as-is (float32, 0–255).
- **Do** normalize ROI coordinates to [0, 1]; keep `batch_idx = 0`.
- **Do** L2-normalize the output embeddings.
- **Don't** apply /255, mean/std, or BGR→RGB outside the graph.
- **Don't** pass pixel-unit ROIs.
- **Don't** consume output rows that correspond to zero-padding.

