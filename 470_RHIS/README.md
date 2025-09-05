# 470_RHIS

A lightweight ROI-based hierarchical instance segmentation model for human detection with knowledge distillation from EfficientNet-based teacher models. The model achieves efficient real-time performance through a two-stage hierarchical architecture and temperature progression distillation techniques.

Detectron/Detectron2, YOLO not used. However, you can always merge this with your favorite object detection model such as YOLO, DEIM, or RT-DETR to have an end-to-end instance segmentation model.

## No pixel expansion

|Images|Images|
|:-:|:-:|
|<img width="640" height="457" alt="000000049759_segmented" src="https://github.com/user-attachments/assets/56699f65-6dc7-430d-a2f3-007669a083c4" />|<img width="640" height="480" alt="000000025593_segmented" src="https://github.com/user-attachments/assets/f8eae65a-865a-47b8-8fdd-b66d484e1118" />|
|<img width="640" height="427" alt="000000050380_segmented" src="https://github.com/user-attachments/assets/c7fcb3dd-91bb-4446-bf55-c9430d507e96" />|<img width="480" height="640" alt="000000074646_segmented" src="https://github.com/user-attachments/assets/d3954e37-546f-4eb3-a8c0-40ee9db518a9" />|

## 1 pixel dilation

|Images|Images|
|:-:|:-:|
|<img width="640" height="457" alt="000000049759_segmented" src="https://github.com/user-attachments/assets/f9ed44cc-ebb7-4464-bf33-0a27a3f2c022" />|<img width="640" height="480" alt="000000025593_segmented" src="https://github.com/user-attachments/assets/c8d9b506-9580-461e-a525-c09f919e3e3b" />|
|<img width="640" height="427" alt="000000050380_segmented" src="https://github.com/user-attachments/assets/8dfc494e-0502-4f46-a323-e3d72d9a990e" />|<img width="480" height="640" alt="000000074646_segmented" src="https://github.com/user-attachments/assets/fead79af-69e1-48ed-ac67-96e287024382" />|

## 160x120 Instance Segmentation Mode

<img width="160" height="120" alt="000000229659_segmented" src="https://github.com/user-attachments/assets/6452429a-31d4-4f8b-afd6-5db5290dcc88" />

## Binary Mask Mode

<img width="427" height="640" alt="000000229849_binary" src="https://github.com/user-attachments/assets/f73ef1b2-36d8-4eb0-b167-06764e05ebe3" />

## Architecture Diagram

```
             ┌─────────────────────────────┐           ┌──────────────────────────────┐
             │       Input RGB Image       │           │             ROIs             │
             │        [B, 3, H, W]         │           │            [N, 5]            │
             └──────────────┬──────────────┘           │ [batch_idx, x1, y1, x2, y2]  │
                            │                          │ (0-1 normalized coordinates) │
                            │                          └──────────────┬───────────────┘
                            │                                         │
             ┌──────────────▼──────────────┐                          │
             │   Pretrained UNet Module    │                          │
             │    (Frozen during training) │                          │
             │   Output: Binary FG/BG      │                          │
             └──────────────┬──────────────┘                          │
                            │                                         │
              ┌─────────────┴─────────────┐                           │
              │                           │                           │
  ┌───────────▼───────────┐   ┌───────────▼──────────┐                │
  │  Binary Mask Output   │   │   Feature Maps       │                │
  │   [B, 1, H, W]        │   │   for ROI Pooling    │                │
  └───────────┬───────────┘   └───────────┬──────────┘                │
              │                           │                           │
              └─────────────┬─────────────┘                           │
                            │◀────────────────────────────────────────┘
            ┌───────────────▼───────────────┐
            │   Dynamic RoI Align           │
            │  Output: [N, C, H_roi, W_roi] │
            └───────────────┬───────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
 ┌────────────▼───────────┐   ┌───────────▼────────────┐
 │      EfficientNet      │   │  Pretrained UNet Mask  │
 │      Encoder           │   │  (for each ROI)        │
 │      (B0/B1/B7)        │   │  [N, 1, H_roi, W_roi]  │
 └────────────┬───────────┘   └───────────┬────────────┘
              │                           │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │  Instance Segmentation    │
              │  Head (UNet V2)           │
              │  - Attention Modules      │
              │  - Residual Blocks        │
              │  - Distance-Aware Loss    │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   3-Class Output Logits   │
              │   [N, 3, mask_h, mask_w]  │
              │   Classes:                │
              │   0: Background           │
              │   1: Target Instance      │
              │   2: Non-target Instances │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   Post-Processing         │
              │   (Optional)              │
              │   - Mask Dilation         │
              │   - Edge Smoothing        │
              └───────────────────────────┘
```

## Cited

https://github.com/PINTO0309/human-instance-segmentation
