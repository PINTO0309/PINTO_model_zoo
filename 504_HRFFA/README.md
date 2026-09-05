# 504_HRFFA

![GitHub](https://img.shields.io/github/license/PINTO0309/High-Angle_Robust_Fast_FaceAlignment?color=2BAF2B) [![DOI](https://zenodo.org/badge/1341204426.svg)](https://doi.org/10.5281/zenodo.22161810)

A training, distillation and ONNX deployment pipeline for **whole-head face alignment that stays robust at extreme head poses while remaining light enough for CPU inference**.

- **Face alignment on head crops, not face crops.** Conventional face-alignment papers assume a tightly cropped face region; this is a special-purpose model that works on a crop of the whole head instead. The architecture is therefore at a considerable disadvantage against paper benchmarks, but the intent is to take in more features by processing the entire head and the background context contained in a small margin at the same time. It is also the practical choice: object detectors localize whole heads far more stably than faces.
- For head crops that include looking up / looking down (measured pitch beyond ±85°), in-plane rotation over the **full 360° roll**, and profile views (yaw ±90°), the models output 68 / 98 / 29 landmarks plus a 3-class visibility label per point (outside the image / occluded / visible).
- The teacher is a DINOv3 ViT-L/16 (320×320); the students are a ViT-T/16 (256 and 96 input) and a PP-HGNetV2-B0 based CNN (256 and 96 input). Students are trained by online distillation from the teacher and exported to ONNX graphs that run in 1.5–12 ms on a CPU.

- Demo

  https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment

<p align="center">
  <img width="46%" alt="image" src="https://github.com/user-attachments/assets/4ded0896-bf84-4319-9357-e25f6fd1807b" />
  <img width="46%" alt="image" src="https://github.com/user-attachments/assets/c63c4212-f62c-4f8b-a9fb-19074eb56cc7" />
</p>

<p align="center"><sub>Left: teacher vitl-320 on nine looking-up × profile tiles with a roll of 0–320° composited onto each tile.<br>Right: student vitt-256 on measured pitch extremes (+86° to −89°) with roll. Predictions only.</sub></p>

- Of the total inference time, 25 ms is spent on CUDA inference for DEIMv2-Wholebody49-S, while the remainder is spent on FaceAlignment inference.

  https://github.com/user-attachments/assets/144e485e-ec5c-460d-9cfb-58334accb92a

- TensorRT BF16

  https://github.com/user-attachments/assets/1226ea6c-7f49-4dcd-8c35-39701a192b5d

- WebGPU + YOLOv9-MIT Demo

  https://github.com/user-attachments/assets/1266c08b-3c2d-4aa6-8804-0427716a56d1

- 360° Yaw Estimation Test

  https://github.com/user-attachments/assets/7eaab9fc-5402-41fb-93f3-49b707422bc9

## 1. Results at a glance

inter-ocular NME %, lower is better; CPU latency on an i9-10900K with onnxruntime 1.22 CPU EP, batch 1.

Checkpoints and ONNX/TFLite(LiteRT): https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment/releases/tag/weights

| Model | Input | Params | GMACs | GFLOPs | CPU ms | WFLW Full | WFLW Pose | 300W Full | COFW |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| [D-ViT (paper)](https://arxiv.org/abs/2411.07167) | 256 | 96.4M | ≈73.15 | ≈146.30 | – | 3.75 | 6.43 | 2.85 | 4.13 |
| vitl-320 (teacher) | 320 | 308.2M | 131.32 | 262.63 | 520 | 1.51 | 2.54 | 1.03 | 1.21 |
| vitt-256 | 256 | 9.0M | 2.05 | 4.09 | 12.4 | 3.36 | 5.64 | 2.66 | 2.76 |
| hg0-256 | 256 | 1.6M | 0.70 | 1.40 | 5.2 | 5.32 | 9.53 | 3.87 | 3.77 |
| vitt-096 | 96 | 9.0M | 0.43 | 0.85 | 4.6 | 5.14 | 8.65 | 3.86 | 3.90 |
| hg0-096 | 96 | 1.6M | 0.13 | 0.26 | 1.5 | 7.46 | 14.71 | 4.77 | 4.99 |

**How to read these numbers (important):** all splits of the real-image datasets — including WFLW test, 300W and COFW test — are used for training. The "official" numbers above therefore measure how well the models fit the training distribution and **must not be compared with published benchmark results**. The only unleaked instrument is `300wlp_val` (head-NME on a held-out 300W-LP subset; values in 050). The full tables — WFLW subsets, FR/AUC, breakdowns by pose (yaw / pitch / roll) and by image degradation, and the D-ViT paper values listed side by side with an explicit non-comparability note — follow in §1.1 (same content as 050 §2). The `D-ViT (paper)` row lists the published benchmark values of arXiv 2411.07167 (Tables 1 / 2, "Ours": 256×256 input, 8 prediction blocks = 96.4M parameters per its Table S1) for format reference only. The paper does not report FLOPs or latency; the GMACs / GFLOPs are our estimate from the [official code](https://github.com/Human3DAIGC/AccurateFacialLandmarkDetection) with its default arguments (8 prediction blocks, 32×32 heatmaps, depth 256, 256×256 input, all 8 stages, batch 1, same `FlopCounterMode` method as the other rows) and are approximate because that configuration has 101.1M parameters, about 5 % more than Table S1. Its COFW NME is normalized by the inter-pupil distance instead of inter-ocular, and its test sets are held out from training, so no ranking against the fit measures above is implied.

### 1.1 Full tables

All tables below are generated from `runs/<run>/eval_best_{official,stratreal,stress,style}.log` by `scripts/make_results_tables.py`. Notes:

- **Absolute values are fit measures** (all real-image splits are in the training data). The `D-ViT (paper)` rows in Tables 1 / 2 are the published test-benchmark values of arXiv 2411.07167 (p.6, "Ours") and are listed for format reference only — the conditions differ, so no ranking is implied.
- **Yaw / pitch / roll**: the models do not output head pose, so pose robustness is expressed as landmark accuracy per pose. Table 3: inter-ocular NME per effective pose bin (6DRepNet-estimated pose plus the applied rotation, 3 sets × 300 images × 16 configurations). Table 4: NME per in-plane roll (exact 360° roll equivariance; a small worst−base means equivariant). Table 5: NME under projective camera pitch ±15/±25° and yaw ±15° perturbations. Table 6: head-NME (normalized by the crop side, not inter-ocular) of the three real sets stratified by 6DRepNet-estimated pose.
- **Table 7 perturbations are applied in crop pixels** (mblur9/21 = 30° motion blur of 9/21 px, warm/cool = channel gains, gamma 0.6/1.6, gray, jpeg30 = JPEG quality 30). Their strength therefore depends on the input resolution: for the 96 models (vitt-096 / hg0-096) a 21 px blur on a 96 px crop corresponds to ~56 px at 256, so their Table 7 rows are not comparable with the 256 models.

#### Table 1. NME (%, inter-ocular, lower is better)
| Model | WFLW<br>Full | <br>Pose | <br>Exp | <br>Ill | <br>Mu | <br>Occ | <br>Blur | COFW<br>Full | 300W<br>Full | <br>Comm | <br>Chal |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| [D-ViT (paper)]( https://arxiv.org/abs/2411.07167) | 3.75 | 6.43 | 3.85 | 4.06 | 3.57 | 4.47 | 4.37 | 4.13 | 2.85 | 2.43 | 4.56 |
| vitl-320 | 1.51 | 2.54 | 1.55 | 1.47 | 1.42 | 1.56 | 1.58 | 1.21 | 1.03 | 0.96 | 1.34 |
| vitt-256 | 3.36 | 5.64 | 3.43 | 3.25 | 3.19 | 3.56 | 3.60 | 2.76 | 2.66 | 2.36 | 3.90 |
| hg0-256 | 5.32 | 9.53 | 5.54 | 5.16 | 5.09 | 6.29 | 6.04 | 3.77 | 3.87 | 3.37 | 5.96 |
| vitt-096 | 5.14 | 8.65 | 5.32 | 4.96 | 5.06 | 5.52 | 5.40 | 3.90 | 3.85 | 3.57 | 5.02 |
| hg0-096 | 7.46 | 14.71 | 7.77 | 7.19 | 8.10 | 8.91 | 8.16 | 4.99 | 4.77 | 4.24 | 6.91 |

#### Table 2. FR10 (%, lower is better) and AUC10 (%, higher is better) on WFLW
| Model | FR10<br>Full | <br>Pose | <br>Exp | <br>Ill | <br>Mu | <br>Occ | <br>Blur | AUC10<br>Full | <br>Pose | <br>Exp | <br>Ill | <br>Mu | <br>Occ | <br>Blur |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| [D-ViT (paper)]( https://arxiv.org/abs/2411.07167) | 1.76 | 8.28 | 1.27 | 1.29 | 1.94 | 3.80 | 2.07 | 63.7 | 40.1 | 62.6 | 64.7 | 64.7 | 57.1 | 58.6 |
| vitl-320 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 84.9 | 74.5 | 84.5 | 85.4 | 85.8 | 84.4 | 84.2 |
| vitt-256 | 0.28 | 1.53 | 0.32 | 0.00 | 0.00 | 0.54 | 0.52 | 66.5 | 43.9 | 65.7 | 67.5 | 68.2 | 64.5 | 64.1 |
| hg0-256 | 7.00 | 32.21 | 5.73 | 5.44 | 6.31 | 13.04 | 9.44 | 49.2 | 17.5 | 46.7 | 50.3 | 50.2 | 42.4 | 42.8 |
| vitt-096 | 4.20 | 23.31 | 4.78 | 2.72 | 2.91 | 5.57 | 3.75 | 49.7 | 20.2 | 47.7 | 51.2 | 49.9 | 46.2 | 47.1 |
| hg0-096 | 17.88 | 67.48 | 18.79 | 14.18 | 20.87 | 26.09 | 22.38 | 36.1 | 5.8 | 31.9 | 37.6 | 34.4 | 28.7 | 30.0 |

#### Table 3. NME (%) by yaw / pitch bin (effective pose bins of pose-stress)
| Model | Yaw<br>0–30 | Yaw<br>30–60 | Yaw<br>60–95 | Pitch<br>−95..−45 | Pitch<br>−45..−15 | Pitch<br>−15..15 | Pitch<br>15..45 | Pitch<br>45..95 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| vitl-320 | 1.26 | 1.71 | 2.04 | 1.76 | 1.37 | 1.30 | 1.29 | 1.85 |
| vitt-256 | 2.94 | 4.10 | 4.68 | 4.07 | 3.20 | 3.06 | 2.99 | 4.43 |
| hg0-256 | 4.21 | 6.35 | 8.14 | 6.26 | 4.63 | 4.46 | 4.35 | 7.51 |
| vitt-096 | 4.49 | 6.21 | 7.37 | 5.93 | 4.82 | 4.77 | 4.54 | 7.06 |
| hg0-096 | 5.69 | 8.70 | 10.85 | 8.02 | 6.17 | 6.21 | 5.79 | 10.07 |
| n | 5933 | 1124 | 143 | 223 | 2503 | 2944 | 1398 | 48 |

#### Table 4. NME (%) by in-plane roll (pose-stress, n=300 per set)
| Model | Set | Roll 0 | 45 | 90 | 135 | 180 | 225 | 270 | 315 | worst−base |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| vitl-320 | wflw | 1.61 | 1.61 | 1.64 | 1.60 | 1.63 | 1.61 | 1.63 | 1.60 | +0.02 |
| | 300w | 1.12 | 1.11 | 1.14 | 1.11 | 1.13 | 1.11 | 1.13 | 1.11 | +0.01 |
| | cofw | 1.29 | 1.30 | 1.31 | 1.30 | 1.30 | 1.30 | 1.30 | 1.29 | +0.02 |
| vitt-256 | wflw | 3.67 | 3.61 | 3.67 | 3.60 | 3.70 | 3.61 | 3.68 | 3.61 | +0.03 |
| | 300w | 2.93 | 2.88 | 2.89 | 2.89 | 2.95 | 2.92 | 2.96 | 2.90 | +0.03 |
| | cofw | 3.01 | 2.96 | 3.02 | 2.97 | 2.98 | 2.96 | 2.98 | 2.97 | +0.01 |
| hg0-256 | wflw | 5.96 | 5.95 | 5.90 | 6.06 | 5.92 | 6.03 | 5.98 | 5.96 | +0.10 |
| | 300w | 4.14 | 4.11 | 4.26 | 4.20 | 4.26 | 4.29 | 4.21 | 4.16 | +0.15 |
| | cofw | 4.12 | 4.08 | 4.06 | 4.06 | 4.03 | 4.11 | 4.12 | 4.06 | +0.00 |
| vitt-096 | wflw | 6.30 | 6.05 | 6.17 | 5.98 | 6.30 | 6.26 | 6.40 | 6.19 | +0.10 |
| | 300w | 4.58 | 4.41 | 4.55 | 4.50 | 5.03 | 5.11 | 5.14 | 4.71 | +0.57 |
| | cofw | 4.74 | 4.61 | 4.72 | 4.49 | 4.73 | 4.49 | 4.56 | 4.55 | +0.00 |
| hg0-096 | wflw | 8.89 | 9.03 | 9.09 | 9.21 | 8.98 | 9.33 | 9.90 | 8.72 | +1.01 |
| | 300w | 5.57 | 5.61 | 5.82 | 6.07 | 6.40 | 6.59 | 6.05 | 5.95 | +1.03 |
| | cofw | 5.84 | 5.79 | 6.05 | 5.71 | 5.76 | 5.51 | 5.68 | 5.73 | +0.21 |

#### Table 5. NME (%) under camera pitch / yaw perturbation (pose-stress, n=300 per set)
| Model | Set | base | cam<br>pitch<br>−25 | <br><br>−15 | <br><br>+15 | <br><br>+25 | cam<br>yaw<br>−15 | <br><br>+15 | <br><br>p+25,y+15 | <br><br>p−25,y−15 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| vitl-320 | wflw | 1.61 | 1.78 | 1.63 | 1.58 | 1.60 | 1.55 | 1.55 | 1.56 | 1.78 |
| | 300w | 1.12 | 1.25 | 1.15 | 1.09 | 1.10 | 1.08 | 1.08 | 1.08 | 1.22 |
| | cofw | 1.29 | 1.36 | 1.29 | 1.26 | 1.29 | 1.23 | 1.24 | 1.25 | 1.35 |
| vitt-256 | wflw | 3.67 | 3.84 | 3.67 | 3.52 | 3.48 | 3.52 | 3.50 | 3.44 | 3.94 |
| | 300w | 2.93 | 3.07 | 2.93 | 2.81 | 2.75 | 2.78 | 2.81 | 2.73 | 3.03 |
| | cofw | 3.01 | 3.12 | 3.01 | 2.95 | 2.92 | 2.90 | 2.90 | 2.92 | 3.12 |
| hg0-256 | wflw | 5.96 | 5.99 | 5.71 | 5.71 | 5.38 | 5.55 | 5.60 | 5.31 | 6.08 |
| | 300w | 4.14 | 4.46 | 4.24 | 4.05 | 3.98 | 3.97 | 4.04 | 3.90 | 4.39 |
| | cofw | 4.12 | 4.27 | 4.11 | 4.02 | 3.96 | 3.98 | 3.97 | 3.99 | 4.23 |
| vitt-096 | wflw | 6.30 | 5.94 | 5.83 | 5.97 | 5.51 | 5.78 | 5.76 | 5.30 | 6.05 |
| | 300w | 4.58 | 4.49 | 4.39 | 4.41 | 4.10 | 4.24 | 4.34 | 4.06 | 4.41 |
| | cofw | 4.74 | 4.53 | 4.49 | 4.45 | 4.18 | 4.33 | 4.42 | 4.20 | 4.47 |
| hg0-096 | wflw | 8.89 | 8.20 | 8.27 | 8.44 | 7.72 | 8.37 | 8.19 | 7.35 | 8.34 |
| | 300w | 5.57 | 5.38 | 5.38 | 5.45 | 4.94 | 5.16 | 5.25 | 4.81 | 5.34 |
| | cofw | 5.84 | 5.62 | 5.57 | 5.67 | 5.28 | 5.35 | 5.53 | 5.13 | 5.48 |

#### Table 6. head-NME (×100) stratified by 6DRepNet-estimated pose (3,696 real images from the three sets)
| Model | mean | Yaw<br>0–30 | Yaw<br>30–60 | Yaw<br>60–95 | Pitch<br>−90..−30 | Pitch<br>−30..−10 | Pitch<br>−10..10 | Pitch<br>10..30 | Pitch<br>30..90 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| vitl-320 | 0.37 | 0.36 | 0.42 | 0.45 | 0.42 | 0.37 | 0.36 | 0.40 | 0.50 |
| vitt-256 | 0.85 | 0.81 | 0.98 | 1.01 | 0.98 | 0.85 | 0.82 | 0.93 | 1.13 |
| hg0-256 | 1.29 | 1.21 | 1.60 | 1.85 | 1.71 | 1.28 | 1.21 | 1.50 | 2.09 |
| vitt-096 | 1.27 | 1.22 | 1.46 | 1.58 | 1.48 | 1.27 | 1.22 | 1.38 | 1.71 |
| hg0-096 | 1.75 | 1.60 | 2.22 | 2.85 | 2.53 | 1.76 | 1.59 | 2.04 | 2.94 |
| n | | 2939 | 672 | 85 | 143 | 1076 | 2136 | 232 | 32 |

#### Table 7. style-shift: NME (%) under image degradations (in parentheses: degradation vs clean, n=300 per set)
| Model | Set | clean | mblur9 | mblur21 | warm | cool | gamma0.6 | gamma1.6 | gray | jpeg30 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| vitl-320 | wflw_test | 1.49 | 1.57 (+5.5%) | 2.09 (+40.4%) | 1.51 (+1.4%) | 1.51 (+1.5%) | 1.50 (+1.1%) | 1.51 (+1.3%) | 1.60 (+7.8%) | 1.55 (+4.5%) |
| | 300w_valid | 1.04 | 1.13 (+8.6%) | 1.46 (+40.6%) | 1.06 (+2.0%) | 1.05 (+1.2%) | 1.06 (+1.8%) | 1.05 (+1.4%) | 1.09 (+4.9%) | 1.08 (+4.3%) |
| | cofw_test | 1.21 | 1.27 (+5.3%) | 1.60 (+32.3%) | 1.22 (+1.3%) | 1.23 (+1.5%) | 1.22 (+1.1%) | 1.22 (+1.2%) | 1.27 (+5.0%) | 1.25 (+3.6%) |
| vitt-256 | wflw_test | 3.26 | 3.32 (+1.8%) | 4.60 (+41.1%) | 3.32 (+1.9%) | 3.33 (+2.1%) | 3.35 (+2.7%) | 3.36 (+3.0%) | 3.48 (+6.6%) | 3.36 (+3.1%) |
| | 300w_valid | 2.65 | 2.73 (+2.6%) | 3.66 (+37.7%) | 2.68 (+1.1%) | 2.69 (+1.3%) | 2.71 (+2.0%) | 2.72 (+2.3%) | 2.75 (+3.7%) | 2.73 (+2.9%) |
| | cofw_test | 2.76 | 2.82 (+2.1%) | 3.51 (+27.2%) | 2.79 (+1.0%) | 2.79 (+1.1%) | 2.83 (+2.4%) | 2.81 (+1.7%) | 2.85 (+3.3%) | 2.81 (+1.8%) |
| hg0-256 | wflw_test | 5.08 | 5.28 (+3.9%) | 8.16 (+60.5%) | 5.14 (+1.2%) | 5.23 (+2.9%) | 5.13 (+0.9%) | 5.29 (+4.2%) | 5.41 (+6.4%) | 5.35 (+5.2%) |
| | 300w_valid | 3.85 | 3.88 (+0.8%) | 5.20 (+35.0%) | 3.90 (+1.2%) | 3.90 (+1.3%) | 3.88 (+0.7%) | 3.92 (+1.7%) | 3.95 (+2.5%) | 3.90 (+1.0%) |
| | cofw_test | 3.79 | 3.90 (+2.9%) | 4.93 (+30.0%) | 3.83 (+1.1%) | 3.83 (+0.9%) | 3.82 (+0.8%) | 3.82 (+0.8%) | 3.91 (+3.2%) | 3.85 (+1.5%) |
| vitt-096 | wflw_test | 5.01 | 6.26 (+25.0%) | 24.80 (+395.2%) | 5.10 (+1.9%) | 5.09 (+1.6%) | 5.13 (+2.4%) | 5.21 (+4.1%) | 5.37 (+7.3%) | 5.36 (+7.0%) |
| | 300w_valid | 3.82 | 4.42 (+15.8%) | 18.31 (+379.4%) | 3.86 (+1.1%) | 3.86 (+1.1%) | 3.90 (+2.1%) | 3.94 (+3.1%) | 3.96 (+3.8%) | 4.04 (+5.8%) |
| | cofw_test | 3.89 | 4.52 (+16.3%) | 15.95 (+310.4%) | 3.95 (+1.8%) | 3.95 (+1.6%) | 3.99 (+2.8%) | 4.00 (+2.8%) | 4.05 (+4.2%) | 4.08 (+5.0%) |
| hg0-096 | wflw_test | 7.01 | 9.18 (+31.0%) | 32.00 (+356.7%) | 7.15 (+2.0%) | 7.30 (+4.2%) | 7.11 (+1.5%) | 7.36 (+5.0%) | 7.77 (+10.9%) | 9.10 (+29.9%) |
| | 300w_valid | 4.64 | 5.68 (+22.5%) | 19.79 (+326.5%) | 4.68 (+1.0%) | 4.74 (+2.1%) | 4.70 (+1.2%) | 4.72 (+1.7%) | 4.80 (+3.5%) | 5.60 (+20.8%) |
| | cofw_test | 5.03 | 5.71 (+13.7%) | 18.03 (+258.9%) | 5.07 (+0.8%) | 5.15 (+2.4%) | 5.04 (+0.3%) | 5.16 (+2.6%) | 5.19 (+3.2%) | 5.74 (+14.2%) |

## 2. Features

- **Unified dataset**: 300W-LP / WFLW / 300W / COFW converted into a single JSONL format (head bbox, visibility, pose, and a per-record `license_tag`). Head boxes and face parts come from DEIMv2-Wholebody49 (Apache-2.0) pseudo labels; extreme poses are reinforced with depth-reprojection synthesis and generated image pools.
- **Geometric augmentation composed into a single homography**: full 360° roll, camera pitch ±25° / yaw ±15°, horizontal flip (with left/right index swap), scale and translation are applied in one warp, and the GT coordinates, visibility and rotation matrix follow exactly. Photometric augmentation (brightness, gamma, grayscale, noise, blur, JPEG quality 35–85, motion blur) is applied on the training side as well.
- **Online teacher → student distillation**: the same augmented crop is rendered at 320 for the teacher and at 256 (or 96) for the student; coordinate, visibility and decoder-token KD are combined with the GT losses. The learning-rate schedule is WSD (warmup → constant → cosine decay over the last epochs); the stable phase can be extended by editing `epochs` and resuming.
- **Evaluation instruments** (`hrffa.train.evaluate`): official (inter-ocular NME / FR@0.1 / AUC@0.1), stratify-real (stratified by 6DRepNet-estimated pose), pose-stress (real images + exact GT geometric transforms to measure equivariance under 360° roll and camera pose), and style-shift (motion blur, color temperature, gamma, grayscale, JPEG).
- **ONNX export** (`hrffa.export.export_onnx`): a static batch-1 graph is optimized with onnxslim → onnxsim (without Gemm fusion), checked for parity against PyTorch, and an N-batch variant (`<stem>_n.onnx`) is derived from it. The batch axis stays the leading axis of every Reshape and no rank-5 tensors are produced.

## 3. Models and the ONNX I/O contract

| | Teacher vitl-320 | Students vitt-256 / vitt-096 | Students hg0-256 / hg0-096 |
|---|---|---|---|
| Backbone | DINOv3 ViT-L/16 (the official implementation is imported at runtime from the torch.hub cache; its code is not vendored) | ViT-T/16 (own implementation with RoPE, initialized from `ckpts/vitt_distill.pt`) | PP-HGNetV2-B0 stages 0–2 (own implementation) + FPN (stride 8) |
| Head | Point-query Transformer decoder (d256, 4 layers) with coordinate regression and visibility | Same (d256, 3 layers) | Same (d128, 2 layers, 4 heads) |
| Input normalization | ImageNet mean/std | center05 `(x/255 − 0.5)/0.5` (no normalization op inside the ONNX graph) | center05, folded into the stem conv (no input op at all) |
| Preset | `clean_v3` | `student_s256_96gb_r2` / `student_s096_96gb_r2` | `student_hg0_wsd` / `student_hg0_s096_wsd` |

ONNX (example for `--scheme ibug68`; `wflw98` / `cofw29` are exported as separate files):

```
# RGB, normalized as above. S = 320 / 256 / 96 (the preset's out_size)
input   images      float32 [N, 3, S, S]
# coordinates relative to the crop (0..1 inside the crop; points outside still get coordinates)
output  points      float32 [N, K, 2]
# visibility logits (0 = outside the image, 1 = occluded, 2 = visible)
output  vis_logits  float32 [N, K, 3]
```

- Crop: a square around the head bbox with a margin of 0.05 of the side length, rendered at S×S (the same geometry as in training and evaluation). The downscaling method (bilinear / area, with or without antialiasing) changes the results by less than ±1%.
- Two files are written: the fixed batch-1 `*.onnx` and `*_n.onnx` with a symbolic batch axis `N`. Both are verified with onnxruntime to agree at batch sizes 1 / 2 / 3.
- Head pose (rotation / roll_bit) is not an output. Pose supervision is not used in training; derive pose with an add-on head or PnP if needed.

## 4. Setup

Prerequisites: Linux, Python 3.13 (pinned in `.python-version`), [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU (torch built for CUDA 12.8; an 8 GB GPU is enough for smoke tests, full training targets a 96 GB class GPU). Dependencies are pinned with `==` in `pyproject.toml` and hash-locked in `uv.lock`.

```bash
git clone https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment.git
cd High-Angle_Robust_Fast_FaceAlignment
uv sync --frozen                                   # default: onnxruntime-gpu 1.22.0 (the validated toolchain)
uv sync --frozen --no-group ort --group tensorrt   # alternative: onnxruntime-gpu 1.26.0 for TensorRT BF16 (trt_bf16_enable)
# NOTE: `uv run` re-syncs the environment with the DEFAULT groups before running, i.e. a plain
# `uv run python ...` silently switches back to 1.22.0. Keep the same group flags on `uv run`
# (or use `uv run --no-sync` / `.venv/bin/python` after the manual sync):
uv run --no-group ort --group tensorrt \
python demo/demo_hrffa_onnx.py \
-v 0 \
-o output_dir \
-d tensorrt

uv run python -m unittest tests/test_model_arms.py
```

### 4.1 Weights (`ckpts/`, none of them are bundled)

| File | Purpose | Source / license |
|---|---|---|
| `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (also vits16 / vitb16 / vith16plus) | Teacher backbone | Distributed by Meta's [DINOv3](https://github.com/facebookresearch/dinov3) (DINOv3 License, no redistribution). The official implementation is fetched into `~/.cache/torch/hub` on first run |
| `vitt_distill.pt` | Initial weights of the ViT-T/16 student | ImageNet-pretrained weights distributed with DEIMv2 (Apache-2.0) |
| `PPHGNetV2_B0_stage1.pth` | Initial backbone weights of the CNN student | ImageNet-pretrained weights distributed with DEIMv2 (Apache-2.0) |
| `data/models/deimv2_*wholebody49*.onnx` | Pseudo labels for head boxes and face parts (DEIMv2-Wholebody49) | Apache-2.0 |
| `data/models/depth_anything_v2_small.onnx`, `sixdrepnet360_*.onnx` | Depth-reprojection synthesis; pose QA and stratification | Follow each distributor's license |

### 4.2 Data (`data/` → `datasets/unified/`)

300W-LP, WFLW, 300W and COFW are distributed for research use and are not bundled. Place them under `data/` and convert them into the unified format (the 300W-LP conversion includes a coordinate offset correction).

```bash
uv run python -m hrffa.dataset.convert --source all --data-root data --out datasets/unified
# run DEIMv2 for head boxes / face parts (cached)
uv run python -m hrffa.dataset.pseudolabel.run   ...
# apply the cache to the unified annotations
uv run python -m hrffa.dataset.pseudolabel.apply ...
uv run python -m hrffa.dataset.materialize --source all
```

See `--help` of each CLI for the arguments.

In addition, I added a large volume of synthetic datasets I created myself to significantly compensate for distributional weaknesses in the publicly available datasets. Everything is mechanically composited using gpt-image-2, combining non-existent people and backgrounds. The GT automatically generated values ​​that appear largely correct based on statistical data. 8,000 images were automatically generated.

<img width="1200" height="1200" alt="grid_3x3" src="https://github.com/user-attachments/assets/f82b1ab1-dd5a-479f-90b9-bb14bc693cab" />

## 5. Training, evaluation and export

```bash
# Teacher (DINOv3 ViT-L/16 @320, WSD, 200 epochs)
uv run python -m hrffa.train.train_teacher --preset clean_v3

# Students (online distillation from teacher clean_v3)
## ViT-T @256
uv run python -m hrffa.train.distill_student --preset student_s256_96gb_r2
## CNN @256
uv run python -m hrffa.train.distill_student --preset student_hg0_wsd
## ViT-T @96 (fine-tuned from the 256 model)
uv run python -m hrffa.train.distill_student --preset student_s096_96gb_r2
# To resume, or to extend the constant-LR phase: edit epochs in the preset and
# add --resume runs/<preset>/<preset>_last.pt

# Evaluation (4 instruments)
B=runs/student_s256_96gb_r2/student_s256_96gb_r2_best_e0449_0.007970.pt
uv run python -m hrffa.train.evaluate \
--ckpt $B \
--preset student_s256_96gb_r2 \
--use-ema \
--official

uv run python -m hrffa.train.evaluate \
--ckpt $B \
--preset student_s256_96gb_r2 \
--use-ema \
--stratify-real

uv run python -m hrffa.train.evaluate \
--ckpt $B \
--preset student_s256_96gb_r2 \
--use-ema \
--pose-stress \
--stress-n 300

uv run python -m hrffa.train.evaluate \
--ckpt $B \
--preset student_s256_96gb_r2 \
--use-ema \
--style-shift \
--style-n 300

# ONNX (fixed batch-1 graph + N-batch variant;
#  parity and N-batch agreement verified with onnxruntime)
uv run python -m hrffa.export.export_onnx \
--ckpt $B \
--preset student_s256_96gb_r2 \
--scheme ibug68 \
--output runs/student_s256_96gb_r2/student_s256_96gb_r2_e449_ibug68.onnx

# Results tables (history/050 format)
uv run python scripts/make_results_tables.py > tables.md
```

Training logs are written to `runs/<preset>/train_log_<preset>.jsonl` (validation metrics per epoch); the best checkpoint is `runs/<preset>/<preset>_best_eXXXX_<val>.pt` and the resume checkpoint is `<preset>_last.pt`. `scripts/run_phase_a.sh` runs the student arms of Phase A back to back.

### ONNX demo (`demo/demo_hrffa_onnx.py`)

A self-contained two-stage demo (only `numpy`, `opencv` and `onnxruntime` are needed): DEIMv2-Wholebody49 detects heads (class 7), each head is cropped exactly as in training / evaluation (square crop of the longer bbox side × 1.1, `--crop_pad 0.05`), and the HRFFA graph predicts the landmarks, which are mapped back to the image. Both models are switchable; the normalization of the alignment model is inferred from its file name (`vitl` → ImageNet mean/std, otherwise center05) and can be forced with `--input_norm`. Fixed batch-1 and N-batch graphs are both accepted (an N-batch graph processes all heads of a frame in one run).

```bash
# images (default models: DEIMv2 dinov3_s wholebody49 + hrffa_vitt_ibug68_1x3x256x256)
uv run python demo/demo_hrffa_onnx.py \
-i images_dir \
-o output_dir

# camera 0 / video file, CUDA EP, other alignment models
uv run python demo/demo_hrffa_onnx.py \
-v 0 \
-o output_dir \
-d cuda

uv run --no-group ort --group tensorrt \
python demo/demo_hrffa_onnx.py \
-v 0 \
-o output_dir \
-d tensorrt \
--head_score_threshold 0.70

uv run python demo/demo_hrffa_onnx.py \
-v input.mp4 \
-o output_dir \
-am data/models/hrffa_hg0_ibug68_Nx3x96x96.onnx

uv run python demo/demo_hrffa_onnx.py \
-i images_dir \
-o output_dir \
-dm data/models/deimv2_hgnetv2_n_wholebody49_ins_s08_maskhead64x2_center_1240query_masks.onnx \
-am data/models/hrffa_vitl_ibug68_1x3x320x320.onnx
```

Both demos can also estimate the **head yaw** with **YawNet**, an original lightweight CNN (SynthYaw project; it borrows only the biternion output representation and the von-Mises loss from the BiternionNet paper, §8.3 — the architecture is our own). The default `data/models/yawnet_distill_064_unified_v6u_kappa_1x3x64x64.onnx` is a YawNet-64 student distilled from a DINOv3 ViT-L teacher; it also outputs `kappa` (the von-Mises concentration = confidence, unused by the demos). Preprocessing: direct 64×64 resize of the head crop, RGB, center05 normalization `((x/255)−0.5)/0.5` → unit (cos θ, sin θ), θ = atan2(sin, cos); the yawpose convention (+90° = viewer-left) is mirrored to the display convention: 0° = facing the camera (image bottom), 90° = image right, 180° = facing away, 270° = image left. It is shown as a ring in the top-right corner for the largest head (`-om/--orientation_model`, `--disable_orientation`, `--disable_orientation_ring`, `--orientation_smooth_tau` — a circular EMA on the (cos θ, sin θ) vector smoothing the ring on video / camera input, display only; default 0 = disabled, e.g. `0.15` to enable — and key `r`; the raw `orientation_deg` is added to the JSON). Camera input is opened at VGA (640×480) in both demos. Options: `-d cpu|cuda|tensorrt` (default: CUDA if available), `--inference_type auto|fp16|bf16|int8` for TensorRT (`auto`, the default, picks BF16 when onnxruntime-gpu ≥ 1.25 — the `tensorrt` dependency group — runs on an Ampere or newer GPU and FP16 otherwise, printing a recommendation to install ≥ 1.25), `--trt_cache_dir data/models/trt_cache` (TensorRT engines are cached per onnxruntime / TensorRT version, precision and GPU under `ort-<ver>_trt-<ver>_<precision>_sm<cc>/`; caches built with another onnxruntime version are deleted at startup so engines are always rebuilt after an upgrade), `--ort_log_level verbose|info|warning|error|fatal` (onnxruntime's global logger; the default `error` hides the TensorRT engine-build warnings such as `Make sure input ... has Int64 binding` and `Could not read timing cache`), `--head_score_threshold 0.5`, `--draw_lines` (ibug68 contour lines), `--disable_bbox`, `--point_radius`, `--save_raw_predictions` (per-image / per-frame JSON with head boxes, points and the 3-class visibility), `--disable_imshow`, `--disable_video_writer`. Keys for video / camera input: `ESC` quit, `b` toggle head boxes, `l` toggle contour lines, `r` toggle the orientation ring. Rendering follows the project convention: predictions only, single color, no visibility color coding.

### Web demo (`demo/web/`, Electron + onnxruntime-web on WebGPU / WASM)

The same two-stage pipeline entirely in the browser engine, derived from the web runtime of [PINTO0309/soma](https://github.com/PINTO0309/soma/tree/main/web): Electron + Vite + React + TypeScript with [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) 1.27.0 (WebGPU execution provider, WASM fallback). Inference runs in a **dedicated Web Worker** by default (`--web-inference-worker main` runs the engines on the UI thread instead), the input is a webcam (VGA), a video file or an image file, and the crop geometry and normalization are identical to the Python demo. The Electron shell (`electron/main.ts`) carries the WebGPU command-line switches of the reference (`ignore-gpu-blocklist`, `enable-zero-copy`, `disable-gpu-sandbox`, `enable-unsafe-webgpu`, `enable-webgpu-developer-features`, `enable-features=Vulkan,WebGPU,WebGPUService,SharedArrayBuffer`, `use-webgpu-adapter=default`), injects the COOP / COEP isolation headers and grants camera access. The page also runs in a plain Chrome / Edge 113+ (`pnpm run dev:browser`); every asset (page, wasm runtime, models) is served from the app's own origin — no CDN.

```bash
# DEIMv2 detectors must be rewritten for onnxruntime-web first (double -> float32, negative axes
# normalized, 1-D MatMul operands made 2-D, IsNaN / IsInf expressed with Equal / Abs / Greater so the
# decoder stays on the GPU, unreachable nodes pruned) and are passed through the same
# optimization pipeline as the HRFFA exports (onnxslim without Gemm fusion -> onnxsim without optimization
# passes -> rank-5 qkv rewrite of the ViT attention -> canonicalized fixed graph); every output is verified
# against the original with onnxruntime (graph optimizations off).
# Output names are shortened and tagged, e.g. deimv2_dinov3_s_wholebody49_boxes_only_webgpu.onnx. Only the
# boxes-only graphs are staged and listed (masks output removed with hrffa.dataset.pseudolabel.make_boxes_only,
# mask head pruned): {hgnetv2_n, dinov3_s, dinov3_x}; the UI defaults to the MIT-licensed
# yolov9_t_wholebody34_0100_1x3x640x640.onnx (see below). The legacy data/models/deimv2_wholebody49_boxes_only.onnx
# (X backbone) is published as deimv2_dinov3_x_wholebody49_boxes_only_webgpu.onnx:
uv run python scripts/onnx_web_compat.py data/models/deimv2_*_boxes_only.onnx --out-dir demo/web/models

cd demo/web
corepack enable pnpm     # Node >= 22; the project pins pnpm@10.33.0 via "packageManager"
pnpm install             # pnpm only (npm / yarn are refused); stages models + wasm, provisions the electron binary
pnpm run fetch:models    # no local models? download the ONNX files from the `weights` release into demo/web/models/
                         # (every file is checked against the SHA-256 pinned in models.lock.json), then pnpm run prepare:assets
pnpm run dev -- --web-inference-worker main   # engines on the UI thread instead of the worker
pnpm run dev -- --devtools                    # open DevTools (detached); closed by default, View > Toggle Developer Tools also works
pnpm run dev             # vite dev server + electron window (dedicated inference worker)
pnpm run start           # production build (dist/ + dist-electron/) and launch electron from file://
pnpm run dev:browser     # vite only: open http://localhost:5274 in Chrome / Edge
```

The detector slot also accepts the raw-head YOLOv9-Wholebody34 exports of [PINTO_model_zoo/455_YOLOv9-Wholebody34](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/455_YOLOv9-Wholebody34) — these are the **MIT-licensed YOLOv9** models trained with [PINTO0309/YOLO](https://github.com/PINTO0309/YOLO) (an MIT re-implementation of YOLOv9 / YOLOv7 / YOLO-RD), not the GPL-3.0 official YOLOv9 code, so they can be redistributed with this demo (`yolov9_{n,t}_wholebody34_0100_1x3x640x640.onnx`, output `[1, 4+34, 8400]`, class 7 = head; both are on the `weights` release and fetched by `fetch:models`; `yolov9_t` is the default detector of the web demo): they are staged as they are — the score threshold and a greedy NMS (IoU 0.5) run in the app, no graph rewrite is needed (≈ 22 ms per frame for yolov9_n / yolov9_t on the RTX 3070 with WebGPU). All ONNX files of the demo (the rewritten `deimv2_*_boxes_only_webgpu.onnx` detectors, the YOLOv9 exports and the `hrffa_*` graphs) are published on the [`weights` release](https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment/releases/tag/weights); `scripts/fetch-models.mjs` downloads them and refuses any file whose SHA-256 differs from `models.lock.json`. `scripts/prepare-assets.mjs` (run by `pnpm install`, `dev` and `build`) stages `demo/web/models/deimv2_*_boxes_only_webgpu.onnx` (the rewritten detectors; masks variants are neither staged nor listed), the YOLOv9 exports, the optional `yawnet_distill_{064,096,128}_unified_v6u_kappa` head-orientation models (from `demo/web/models/` or `data/models/`; not part of the `weights` release, so place the file there yourself; selectable in the UI and drawn as a top-right ring; the `Smooth the ring` checkbox — off by default — smooths it with a circular EMA, τ = 150 ms, on video / camera input) and `data/models/hrffa_*.onnx` (the HRFFA graphs run unmodified; files above 300 MB such as the ViT-L teacher are skipped unless `HRFFA_WEB_INCLUDE_LARGE=1`), and copies the onnxruntime-web wasm files next to them. URL parameters: `?backend=wasm|webgpu`, `?worker=dedicated|main`, `?detector=<file>`, `?aligner=<file>`, `?orientation=<file>|off`, `?image=<same-origin URL>` or `?video=<same-origin URL>` with `&autostart=1`.

Measured in Chrome 151 / Electron 43 on an RTX 3070 (WebGPU EP, second run on a still image; WASM = single thread): DEIMv2 boxes-only detectors ≈ 60–65 ms per frame in the app including preprocessing (dinov3_s graph alone ≈ 50 ms; after the rewrites only the 9 int64 index ops of the post-processor remain on the CPU, the rest of the graph runs on WebGPU), vitt-256 ≈ 20 ms per head (N-batch: 9 heads in 144 ms), hg0-256 ≈ 10 ms per head (9 heads in 46 ms); WASM: hgnetv2_n ≈ 610 ms, hg0-256 ≈ 55 ms per head.

Supply-chain hardening (same policy as the reference project this demo is derived from): every dependency is pinned to an exact version and `pnpm-lock.yaml` is committed with integrity hashes (the resolution was taken over unchanged from the reference project — the lockfile is a strict subset of it); `minimumReleaseAge: 10080` in `pnpm-workspace.yaml` refuses any package version published less than 7 days ago; dependency build scripts are blocked except `electron` (official binary download verified against SHASUMS256) and `esbuild`; installs with npm / yarn are refused by a `preinstall` guard; the page ships a strict Content-Security-Policy and loads nothing from third-party hosts.

### Configuration (`configs/*.yaml`)

Each preset stores **only the differences** from the defaults of `TrainConfig` (`src/hrffa/train/config.py`). `_base_:` inherits another preset (chained, last-wins merge, cycle detection; unknown keys are errors).

| Preset | Contents |
|---|---|
| `clean_v3` | Deployment teacher (DINOv3 ViT-L/16 @320, initialized from clean_v2, WSD 200) |
| `student_s_base` → `student_s256_96gb` | Shared student settings (teacher clean_v3, crop pad 0.05, KD 0.5/0.5/0.2, WSD 250/50, EMA 0.999) |
| `student_s256_96gb_r2` | ViT-T @256, second round from the old A1 best (WSD 450) |
| `student_s096_96gb_r2` | ViT-T @96, fine-tuned from the r2 best |
| `student_hg0_wsd` / `student_hg0_s096_wsd` | CNN student @256 / @96 |
| `abl_*` | Ablations |
| `smoke_s_8gb` | Smoke test on an 8 GB GPU |

## 6. Repository layout

```
configs/            training presets (YAML with _base_ inheritance)
src/hrffa/
  dataset/          unified-dataset conversion, pseudo labeling, augmentation, QA tools (converters / pseudolabel / augment / qa / selftrain)
  data/             Dataset classes for training and evaluation (apply geometric / photometric augmentation)
  model/            teacher.py (the model), vit_tiny.py, hgnetv2.py, export_modules.py (export-time MHA), popos.py, losses.py
  train/            train_teacher.py, distill_student.py, evaluate.py, config.py
  export/           export_onnx.py, nbatch.py (fixed batch-1 → N-batch conversion and graph rewrites)
scripts/            run_phase_a.sh (run the student arms back to back), make_results_tables.py, onnx_web_compat.py (DEIMv2 graphs -> onnxruntime-web)
demo/               demo_hrffa_onnx.py (ONNX demo: DEIMv2-Wholebody49 head detection + HRFFA landmarks; images / video / camera)
  web/              Electron / browser demo (Vite + React + onnxruntime-web, WebGPU / WASM, dedicated worker; pinned pnpm lockfile)
tests/              unit tests (model, export, N-batch conversion, dataset conversion)
history/            050_results_tables.md (results tables) and assets/050/ (README figures)
ckpts/ data/ datasets/ runs/ onnx/   weights, raw data, unified data and training outputs (not tracked by git)
```

## 7. License

- Code: [MIT License](LICENSE) (Copyright (c) 2026 Katsuya Hyodo).
- The datasets (300W-LP / WFLW / 300W / COFW) and the generated / self-training data are not bundled; follow their research-use distribution terms. The unified format carries a per-record `license_tag` so that records can be filtered by license.
- Weights: DINOv3 (Meta, DINOv3 License), vitt_distill and PP-HGNetV2 (DEIMv2, Apache-2.0) are neither bundled nor redistributed. Check the derived-work terms of these sources before distributing trained HRFFA weights or ONNX files.

## 8. Citation and acknowledgements

### 8.1 Citing this repository

```bibtex
@software{hyodo2026hrffa,
  author    = {Katsuya Hyodo},
  title     = {HRFFA: High-Angle Robust Fast FaceAlignment},
  month     = {aug},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22161811},
  url       = {https://doi.org/10.5281/zenodo.22161811},
}
```

### 8.2 Backbones, pretrained weights and tools

**DINOv3** — teacher backbone (ViT-L/16). The official implementation is imported at runtime from the torch.hub cache and the weights are distributed under the DINOv3 License. Paper: https://arxiv.org/abs/2508.10104 · Code: https://github.com/facebookresearch/dinov3

```bibtex
@misc{simeoni2025dinov3,
  title         = {{DINOv3}},
  author        = {Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year          = {2025},
  eprint        = {2508.10104},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2508.10104}
}
```

**DEIMv2** — source of the ImageNet-pretrained PP-HGNetV2-B0 weights used by the CNN student (Apache-2.0). Paper: https://arxiv.org/abs/2509.20787 · Code: https://github.com/Intellindust-AI-Lab/DEIMv2

```bibtex
@article{huang2025deimv2,
  title   = {Real-Time Object Detection Meets {DINOv3}},
  author  = {Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
  journal = {arXiv preprint arXiv:2509.20787},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.20787}
}
```

**DEIMv2-Wholebody49** — pseudo labels for head boxes and face parts (Apache-2.0). https://github.com/PINTO0309/PINTO_model_zoo/tree/main/488_DEIMv2-Wholebody49

```bibtex
@software{DEIMv2-Wholebody49,
  author = {Katsuya Hyodo},
  title  = {Unified multi-task model for detection, pose estimation, and instance segmentation. 49 classes.},
  url    = {https://github.com/PINTO0309/PINTO_model_zoo/tree/main/488_DEIMv2-Wholebody49},
  year   = {2026},
  month  = {05},
  doi    = {10.5281/zenodo.10229410}
}
```

**YOLOv9-Wholebody34 (MIT-licensed YOLOv9)** — optional head detector of the web demo (`yolov9_{n,t}_wholebody34_0100_1x3x640x640.onnx`, PINTO_model_zoo #455). The models are trained and exported with [PINTO0309/YOLO](https://github.com/PINTO0309/YOLO), an MIT-licensed re-implementation of YOLOv9 / YOLOv7 / YOLO-RD; the GPL-3.0 official YOLOv9 code is not used. Original paper: https://arxiv.org/abs/2402.13616

```bibtex
@software{PINTO0309_YOLO,
  author = {Katsuya Hyodo},
  title  = {An {MIT} License of {YOLOv9}, {YOLOv7}, {YOLO-RD}},
  url    = {https://github.com/PINTO0309/YOLO},
  year   = {2025}
}

@software{YOLOv9-Wholebody34,
  author = {Katsuya Hyodo},
  title  = {{YOLOv9-Wholebody34}: whole-body 34-class detector trained with the {MIT}-licensed {YOLOv9}},
  url    = {https://github.com/PINTO0309/PINTO_model_zoo/tree/main/455_YOLOv9-Wholebody34},
  year   = {2025}
}

@article{wang2024yolov9,
  title   = {{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
  author  = {Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
  journal = {arXiv preprint arXiv:2402.13616},
  year    = {2024},
  url     = {https://arxiv.org/abs/2402.13616}
}
```

**6DRepNet / 6DRepNet360** — head-pose estimator used for pose QA and for the pose-stratified evaluation (ONNX from PINTO_model_zoo #423). Papers: https://arxiv.org/abs/2202.12555 · https://arxiv.org/abs/2309.07654

```bibtex
@inproceedings{hempel2022sixdrepnet,
  title     = {{6D} Rotation Representation For Unconstrained Head Pose Estimation},
  author    = {Hempel, Thorsten and Abdelrahman, Ahmed A. and Al-Hamadi, Ayoub},
  booktitle = {IEEE International Conference on Image Processing (ICIP)},
  year      = {2022},
  url       = {https://arxiv.org/abs/2202.12555}
}

@article{hempel2023sixdrepnet360,
  title   = {Towards Robust and Unconstrained Full Range of Rotation Head Pose Estimation},
  author  = {Hempel, Thorsten and Abdelrahman, Ahmed A. and Al-Hamadi, Ayoub},
  journal = {arXiv preprint arXiv:2309.07654},
  year    = {2023},
  url     = {https://arxiv.org/abs/2309.07654}
}
```

**Depth Anything V2** — monocular depth used for the depth-reprojection synthesis of extreme pitch (ONNX: onnx-community/depth-anything-v2-small, Apache-2.0). Paper: https://arxiv.org/abs/2406.09414

```bibtex
@inproceedings{yang2024depthanythingv2,
  title     = {Depth Anything {V2}},
  author    = {Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.09414}
}
```

### 8.3 Methods referenced (reimplemented from the papers; no code was copied)

**D-ViT** — the results tables in §1.1 follow the layout of its Tables 1 / 2; its published values are listed for format reference only. Paper: https://arxiv.org/abs/2411.07167

```bibtex
@inproceedings{dang2025dvit,
  title     = {Cascaded Dual Vision Transformer for Accurate Facial Landmark Detection},
  author    = {Dang, Ziqiang and Li, Jianfang and Liu, Lin},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2411.07167}
}
```

**POPoS** — distance-map + multilateration decoding (ablation arm D8). Paper: https://arxiv.org/abs/2410.09583

```bibtex
@inproceedings{xiang2025popos,
  title     = {{POPoS}: Improving Efficient and Robust Facial Landmark Detection with Parallel Optimal Position Search},
  author    = {Xiang, Chong-Yang and He, Jun-Yan and Cheng, Zhi-Qi and Wu, Xiao and Hua, Xian-Sheng},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2410.09583}
}
```

**LESA** — local term added in parallel to self-attention (ablation arm D7). Paper: https://arxiv.org/abs/2107.05637

```bibtex
@article{yang2021lesa,
  title   = {Locally Enhanced Self-Attention: Combining Self-Attention and Convolution as Local and Context Terms},
  author  = {Yang, Chenglin and Qiao, Siyuan and Kortylewski, Adam and Yuille, Alan},
  journal = {arXiv preprint arXiv:2107.05637},
  year    = {2021},
  url     = {https://arxiv.org/abs/2107.05637}
}
```

**BiternionNet** — YawNet, the head-yaw model of the §5 demos, is an original architecture that borrows only this paper's biternion output representation (unit (cos θ, sin θ) regression) and von-Mises loss. Paper: https://lucasb.eyer.be/academic/biternions/biternions_gcpr15.pdf, original implementation (MIT): https://github.com/lucasb-eyer/BiternionNet

```bibtex
@inproceedings{beyer2015biternion,
  title     = {Biternion Nets: Continuous Head Pose Regression from Discrete Training Labels},
  author    = {Beyer, Lucas and Hermans, Alexander and Leibe, Bastian},
  booktitle = {German Conference on Pattern Recognition (GCPR)},
  year      = {2015},
  url       = {https://lucasb.eyer.be/academic/biternions/biternions_gcpr15.pdf}
}
```

**PersonViT** — the fixed batch-1 → N-batch ONNX conversion follows the approach of its `export_onnx.py`. https://github.com/PINTO0309/PersonViT

### 8.4 Datasets

**300W-LP** (used with a coordinate offset correction; also the source of the unleaked `300wlp_val` set). Paper: https://arxiv.org/abs/1511.07212

```bibtex
@inproceedings{zhu2016face,
  title     = {Face Alignment Across Large Poses: A {3D} Solution},
  author    = {Zhu, Xiangyu and Lei, Zhen and Liu, Xiaoming and Shi, Hailin and Li, Stan Z.},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016},
  url       = {https://arxiv.org/abs/1511.07212}
}
```

**WFLW**. Paper: https://arxiv.org/abs/1805.10483

```bibtex
@inproceedings{wu2018wflw,
  title     = {Look at Boundary: A Boundary-Aware Face Alignment Algorithm},
  author    = {Wu, Wayne and Qian, Chen and Yang, Shuo and Wang, Quan and Cai, Yici and Zhou, Qiang},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2018},
  url       = {https://arxiv.org/abs/1805.10483}
}
```

**300W**. https://ibug.doc.ic.ac.uk/resources/300-W/

```bibtex
@inproceedings{sagonas2013300w,
  title     = {300 Faces in-the-Wild Challenge: The First Facial Landmark Localization Challenge},
  author    = {Sagonas, Christos and Tzimiropoulos, Georgios and Zafeiriou, Stefanos and Pantic, Maja},
  booktitle = {IEEE International Conference on Computer Vision Workshops (ICCVW)},
  year      = {2013}
}
```

**COFW**. http://www.vision.caltech.edu/xpburgos/ICCV13/

```bibtex
@inproceedings{burgosartizzu2013cofw,
  title     = {Robust Face Landmark Estimation under Occlusion},
  author    = {Burgos-Artizzu, Xavier P. and Perona, Pietro and Doll{\'a}r, Piotr},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2013}
}
```

### 8.5 Acknowledgements

- Meta AI for DINOv3, the DEIMv2 authors for the detector and the PP-HGNetV2 weights (originally from PaddlePaddle's PaddleClas / PaddleDetection), the authors of 6DRepNet360 and Depth Anything V2 for the tools used in data QA and synthesis, and the dataset authors of 300W-LP, WFLW, 300W and COFW.
- The generated head-image pools used to reinforce extreme poses were produced with OpenAI's image generation API; the synthetic data is not redistributed.
- The ONNX post-processing know-how (batch-axis preservation, N-batch derivation, rank-5 elimination) builds on the author's PersonViT export pipeline and the [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) conventions.
