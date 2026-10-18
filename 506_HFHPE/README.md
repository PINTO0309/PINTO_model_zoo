# 506_HFHPE: High-Angle Robust Fast Head Pose Estimation

![GitHub](./assets/mit.svg) [![DOI](./assets/zenodo.23252206.svg)](https://doi.org/10.5281/zenodo.23252205) [![Ask DeepWiki](./assets/badge.svg)](https://deepwiki.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation)

A training, distillation and ONNX deployment pipeline for **one-pass yaw / pitch / roll head-pose estimation that stays robust at extreme head poses and under heavy occlusion while remaining light enough for CPU inference**.

- **Head pose on whole-head crops.** HFHPE runs on head boxes produced by an object detector. The whole head is localized stably regardless of its orientation, so the pipeline keeps estimating through profiles, strong back-tilt, the back of the head, and heavy occlusion.
- **Full-surround output.** Yaw covers the full 0–360° circle (including the back of the head), pitch covers −120…+120° (beyond +90° = back-tilt), and roll covers the full 360° of in-plane rotation. Every angle is regressed as a biternion (unit `cos/sin`) with a von Mises concentration κ as a per-angle confidence.
- **One-pass integrated graph.** A tiny roll branch estimates the in-plane rotation, the crop is derotated inside the ONNX graph (GridSample), and the yaw/pitch body then runs on the upright crop — one input, one run, six trigonometric outputs plus three confidences.
- The teacher is a DINOv3 ViT-L/16 (320×320); the students are a ViT-T/16, a PP-HGNetV2-B0 based CNN and a from-scratch MBConv CNN (all 64×64 input). Students are trained by online distillation from the teacher and exported to verified ONNX graphs.
- **Avoiding License Contamination Through the Use of Non-Commercial Data.** Train the model without using any data restricted to non-commercial use. This is to avoid wasting time on pointless debates regarding whether the generated model is non-commercial. As a trade-off for that advantage, it is not possible to make a precise performance comparison with all previous paper implementations benchmarked using datasets such as 300W-LP, AFLW200-3D, BIWI, and CMU-Panoptic. It should be noted that for the 300W-LP, AFLW200-3D, BIWI, and CMU-Panoptic datasets, benchmark figures derived from existing implementations do not accurately reflect real-world performance due to significant deficiencies in head pose distribution, real-world head occlusion, and background context diversity.
- Training data for yaw/pitch is **synthetic by gpt-image-2, plus 1,681 of the author's own self-portraits** (the SynthYawPitchPose dataset: 51,194 synthetic whole-head crops, 18.7k of them with large accessories such as masks, sunglasses and hats, every label reviewed by eye and 35.6k labels corrected by hand; plus 1,681 self-portrait crops, sources `synthetic_real_001` / `synthetic_real_002`, labeled with the teacher's pseudo-labels and corrected by hand where needed); the roll branch additionally adapts to real photographs whose roll ground truth is generated mechanically by rotation — no external pose label is ever used as a training target.
- **Roll is decoupled from yaw/pitch, by design.** Synthesizing a head at a prescribed yaw and pitch while keeping the composition intact is not difficult once the right generation conditions are met, but roll is hard to express, and a roll twist skews the directions along which yaw and pitch are estimated. Only yaw and pitch are therefore prescribed strictly: every yaw/pitch image is generated upright (roll assumed to be 0°), and roll is learned on its own from upright crops rotated in-plane by a known angle, which makes the roll ground truth exact.
- **Occlusion diversity is generated, not only erased.** On top of the strictly prescribed yaw and pitch, the dataset emphasizes the diversity of head occlusion by accessories that are likely to occur in the real world. It contains a large number of pose–accessory combinations that hardly exist in public datasets, so that occlusion robustness does not rest solely on simplistic, unrealistic preprocessing such as Random Erasing or Cutout.

> [!NOTE]
> Approximately 24 ms of the inference time is spent on object detection by DEIMv2. The head detection model does not need to be DEIMv2; a lightweight object detection model such as YOLO is sufficient.

- CUDA12.8 + RTX3070 + DEIMv2-S + HFHPE-ViTT 64x64 + webcam realtime

  https://github.com/user-attachments/assets/145bd7ce-e9a9-4e54-b5dc-90e9bc217bb1

- Comparison of behavior with implementations from previous papers.

  |6DRepNet360<br>Contains non-commercial data contamination|HFHPE<br>No non-commercial data contamination|
  |:-:|:-:|
  |<img width="360" height="270" alt="image" src="https://github.com/user-attachments/assets/2f2ce544-6424-43fe-8a66-0d1e3337168e" />|<img width="360" height="270" alt="image" src="https://github.com/user-attachments/assets/c0094d29-4d6d-4fdf-ac05-00cd8d8ba62d" />|

- ELP-USB500W05G-BL170 (Fisheye). Tested under brutal conditions: worst-case exposure, intense backlight, and a head size of around 20x20 pixels.
  - ELP cameras have very poor automatic exposure adjustment; consequently, if there are areas with high light levels in the scene, areas with low light levels become extremely dark.
  - The color quality isn't as good as that of the cheap USB camera I bought for $10 five years ago; the image has an overall yellowish tint.

  https://www.amazon.co.jp/dp/B01DEWOOUM

  https://github.com/user-attachments/assets/297b3d26-586a-4162-b722-e306480f143f

- ELP-USB500W05G-BL170 (Fisheye) + Tested in an ultra-low-lux environment. Under such adverse conditions, there are too few cues for yaw, causing the system to fail.

  <img width="640" height="480" alt="HFHPE ONNX demo_screenshot_10 10 2026" src="https://github.com/user-attachments/assets/6f0458e8-7e95-4ccd-b987-bb51f2d71eb6" />

- VGA USB cam + Exposure time: 205 ms. The test was conducted with all lights turned off, except for the display. Ambient light from the room's windows was completely blocked out using blackout curtains.

  https://github.com/user-attachments/assets/4982db26-7159-47f6-bdc2-2a93336ba703

  |Actual brightness|Exposure time: 205 ms|
  |:-:|:-:|
  |<img width="640" height="480" alt="mpv-shot0005" src="https://github.com/user-attachments/assets/bca402d9-da2a-4021-bab8-92191098876c" />|<img width="640" height="480" alt="mpv-shot0006" src="https://github.com/user-attachments/assets/17bac940-66db-4b82-bd9a-7082c2d85cb9" />|

- Samples of the lower and upper limits of illuminance during learning

  <img width="640" alt="sheet" src="https://github.com/user-attachments/assets/684efbed-1bd1-4288-88b6-f3e2d7ed3b9e" />

- **[October 17, 2026]** As shown in the figure below, we manually assigned ground-truth labels to approximately 1,000 low-light images of heads and performed fine-tuning for low-light conditions.

  <img width="640" alt="image" src="https://github.com/user-attachments/assets/4dbae4b8-d1bc-48dd-9e15-0007b838af98" />

- DEIMv2-S score threshold>=0.35 + ViTT 128x128 - Illuminance variation test. CAMERA: JETAKu c33 VGA

  <img width="3272" height="3268" alt="brightness_extended_5x5" src="https://github.com/user-attachments/assets/c59f6e95-6ac9-4ab9-813e-6056fe148e62" />

- DEIMv2-S score threshold>=0.35 + DINOv3 320x320 - Testing using a thermal camera. CAMERA: HIKMICRO Pocket2
  - Since the training data does not include a single image from a thermal camera, the head orientation estimation becomes highly inaccurate.
  - Since the object detection component is functioning correctly, it is likely that the head orientation estimation model could also achieve high accuracy if fine-tuned using additional training data consisting of thermal camera footage.

  https://github.com/user-attachments/assets/30357e8a-0323-409e-9b1f-7bd2014d134c

## 1. Results at a glance

Validation MAAE (mean absolute angular error, degrees, lower is better) of the current generation (all four bodies retrained). Params and GMACs are counted on the exported ONNX graphs (`scripts/count_macs_onnx.py`; GFLOPs = 2 × GMACs). CPU ms = average latency of the batch-1 graph on an i9-10900K with onnxruntime 1.22 CPU EP, measured with [sit4onnx](https://github.com/PINTO0309/sit4onnx).

The weights and ONNX/TFLite files can be downloaded from the [weights](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/tag/weights) tag in the releases.

This is a benchmark based on the SynthYawPitchPose dataset; it does not utilize any data—such as 300W-LP, AFLW200-3D, BIWI, or CMU-Panoptic—that is of extremely low quality.

**Yaw / pitch body alone** (`<run>_kappa_*.onnx`, `hfhpe_dinov3_yawpitch_*.onnx` for the teacher; no roll, input assumed upright):

| Body | Backbone | Input | Params<br>(M) | GMACs<br>GFLOPs | CPU<br>(ms) | MAAE<br>yaw | <br>pitch |
|---|---|--:|--:|--:|--:|--:|--:|
| dinov3-320 | ViT-L/16 | 320x320 | 304.2 | 130.7<br>261.4 | 352 | 0.25 | 0.21 |
| [vitt-128](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/vitt_distill_128_yp_unified_v12.tar.gz) | ViT-T/16 | 128x128 | 5.6 | 0.374<br>0.748 | 3.3 | 0.53 | 0.49 |
| [vitt-064](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/vitt_distill_064_yp_unified_v12.tar.gz) | ViT-T/16 | 64x64 | 5.6 | 0.094<br>0.188 | 1.8 | 0.76 | 0.67 |
| [hgnetv2-128](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/hgnetv2_distill_128_yp_unified_v12.tar.gz) | PP-HGNetV2-B0 | 128x128 | 1.9 | 0.156<br>0.312 | 1.4 | 1.83 | 1.71 |
| [hgnetv2-064](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/hgnetv2_distill_064_yp_unified_v12.tar.gz) | PP-HGNetV2-B0 | 64x64 | 1.9 | 0.039<br>0.078 | 0.7 | 2.34 | 2.02 |
| [yawnet-128](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/yawnet_distill_128_yp_unified_v12.tar.gz) | MBConv | 128x128 | 0.76 | 0.050<br>0.101 | 1.1 | 5.22 | 7.54 |
| [yawnet-064](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/yawnet_distill_064_yp_unified_v12.tar.gz) | MBConv | 64x64 | 0.76 | 0.013<br>0.026 | 0.7 | 5.55 | 7.55 |

**Integrated one-pass HFHPE** (`hfhpe_*.onnx` = roll branch + in-graph derotation + the body above; the roll branch adds 0.77M params):

| Model | Backbone | Input | Params<br>(M) | GMACs<br>GFLOPs | CPU<br>(ms) | MAAE<br>yaw | <br>pitch | <br>roll |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| hfhpe_dinov3 | ViT-L/16 | 320x320 | 305.1 | 130.7<br>261.5 | 354 | 0.25 | 0.21 | [2.83](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/hfhpe_roll_128_unified_kd_vitl_canvas4_w100.tar.gz) |
| [hfhpe_vitt](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/vitt_distill_128_yp_unified_v12.tar.gz) | ViT-T/16 | 128x128 | 6.4 | 0.425<br>0.849 | 4.9 | 0.53 | 0.49 | [2.83](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/hfhpe_roll_128_unified_kd_vitl_canvas4_w100.tar.gz) |
| [hfhpe_vitt](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/vitt_distill_064_yp_unified_v12.tar.gz) | ViT-T/16 | 64x64 | 6.4 | 0.107<br>0.214 | 2.5 | 0.76 | 0.67 | [3.28](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/hfhpe_roll_064_unified_kd_vitl_canvas4_w100.tar.gz) |
| [hfhpe_hgnetv2](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/hgnetv2_distill_128_yp_unified_v12.tar.gz) | PP-HGNetV2-B0 | 128x128 | 2.6 | 0.207<br>0.413 | 3.2 | 1.83 | 1.71 | [2.83](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/hfhpe_roll_128_unified_kd_vitl_canvas4_w100.tar.gz) |
| [hfhpe_hgnetv2](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/hgnetv2_distill_064_yp_unified_v12.tar.gz) | PP-HGNetV2-B0 | 64x64 | 2.6 | 0.052<br>0.104 | 1.4 | 2.34 | 2.02 | [3.28](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/hfhpe_roll_064_unified_kd_vitl_canvas4_w100.tar.gz) |
| [hfhpe_yawnet](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/yawnet_distill_128_yp_unified_v12.tar.gz) | MBConv | 128x128 | 1.6 | 0.101<br>0.202 | 2.6 | 5.22 | 7.54 | [2.83](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/hfhpe_roll_128_unified_kd_vitl_canvas4_w100.tar.gz) |
| [hfhpe_yawnet](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/yawnet_distill_064_yp_unified_v12.tar.gz) | MBConv | 64x64 | 1.5 | 0.026<br>0.051 | 1.4 | 5.55 | 7.55 | [3.28](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights/hfhpe_roll_064_unified_kd_vitl_canvas4_w100.tar.gz) |

**Accuracy by pose bin (body-only graphs, validation set).** MAAE in degrees, each cell is `yaw / pitch`; the bins are the ground-truth |yaw| (frontal 0° to the back of the head 180°, left and right folded together) and the ground-truth pitch. The last row is the number of validation crops per bin (`scripts/eval_pose_bins.py`).

| Model | \|yaw\|<br>0–30 | <br>30–60 | <br>60–90 | <br>90–120 | <br>120–150 | <br>150–180 |
|---|--:|--:|--:|--:|--:|--:|
| dinov3-320 | 0.23 / 0.25 | 0.26 / 0.19 | 0.25 / 0.20 | 0.23 / 0.20 | 0.29 / 0.23 | 0.25 / 0.21 |
| vitt-128 | 0.53 / 0.54 | 0.53 / 0.44 | 0.57 / 0.45 | 0.44 / 0.44 | 0.56 / 0.54 | 0.50 / 0.47 |
| vitt-064 | 0.77 / 0.73 | 0.77 / 0.60 | 0.76 / 0.61 | 0.65 / 0.61 | 0.83 / 0.75 | 0.73 / 0.66 |
| hgnetv2-128 | 2.07 / 1.94 | 1.74 / 1.61 | 1.65 / 1.74 | 1.52 / 1.51 | 2.02 / 1.78 | 1.93 / 1.62 |
| hgnetv2-064 | 2.39 / 2.22 | 2.22 / 1.74 | 2.16 / 2.03 | 2.03 / 1.79 | 2.56 / 2.16 | 2.58 / 2.00 |
| yawnet-128 | 3.31 / 7.39 | 6.43 / 8.32 | 5.15 / 7.16 | 5.31 / 7.04 | 5.88 / 7.86 | 5.39 / 7.88 |
| yawnet-064 | 3.74 / 7.24 | 6.55 / 8.11 | 5.39 / 7.42 | 5.42 / 6.96 | 6.23 / 7.92 | 6.16 / 7.78 |
| n | 789 | 434 | 1036 | 909 | 1349 | 630 |

| Model | pitch<br>−120..−90 | <br>−90..−60 | <br>−60..−30 | <br>−30..0 | <br>0..30 | <br>30..60 | <br>60..90 | <br>90..120 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| dinov3-320 | 0.21 / 0.32 | 0.28 / 0.25 | 0.25 / 0.20 | 0.25 / 0.19 | 0.25 / 0.21 | 0.25 / 0.25 | 0.26 / 0.29 | 0.21 / 0.33 |
| vitt-128 | 0.53 / 0.69 | 0.57 / 0.54 | 0.57 / 0.48 | 0.50 / 0.46 | 0.51 / 0.50 | 0.55 / 0.49 | 0.49 / 0.53 | 0.47 / 0.54 |
| vitt-064 | 0.82 / 0.95 | 0.86 / 0.70 | 0.74 / 0.65 | 0.73 / 0.66 | 0.77 / 0.68 | 0.77 / 0.63 | 0.66 / 0.85 | 0.71 / 0.83 |
| hgnetv2-128 | 2.08 / 2.18 | 1.90 / 1.60 | 1.84 / 1.59 | 1.79 / 1.72 | 1.82 / 1.72 | 1.72 / 1.74 | 2.35 / 2.35 | 2.16 / 2.25 |
| hgnetv2-064 | 2.44 / 2.66 | 2.51 / 2.08 | 2.38 / 1.96 | 2.24 / 2.03 | 2.32 / 1.90 | 2.35 / 2.20 | 2.61 / 2.39 | 2.48 / 2.47 |
| yawnet-128 | 3.21 / 9.36 | 5.36 / 8.30 | 5.39 / 8.13 | 5.13 / 7.21 | 5.32 / 6.38 | 5.33 / 9.36 | 4.76 / 9.90 | 3.45 / 7.98 |
| yawnet-064 | 4.92 / 10.37 | 6.01 / 8.21 | 5.65 / 8.17 | 5.52 / 7.13 | 5.53 / 6.31 | 5.52 / 9.65 | 4.98 / 9.92 | 4.19 / 8.48 |
| n | 36 | 456 | 889 | 1566 | 1471 | 537 | 129 | 63 |

Observations: the ViT-based bodies are flat across the full yaw circle (side and back views cost nothing extra), and their pitch error grows only at the extreme bins beyond ±90°, which hold few samples. The MBConv student is clearly weaker outside the frontal bin for yaw and above 30° / below −30° for pitch, i.e. it is the capacity, not the pose coverage of the data, that limits the smallest model.

**Style shift (body-only graphs, validation set, n=5,147):** The degradation set of HRFFA's Table 7, applied in crop pixels after the resize to the model input: mblur9 / mblur21 = linear motion blur of 9 / 21 px at 30°, warm / cool = channel gains (B,G,R = 0.8,1.0,1.2 and 1.2,1.0,0.8), gamma 0.6 / 1.6, gray, jpeg30 = JPEG quality 30. Because the degradations are applied at the input resolution their strength depends on it (a 21 px blur on a 64 px input corresponds to about 105 px at 320), so the 64 px rows are not comparable with the 128 / 320 px rows. In parentheses: change against clean (`scripts/eval_style_shift.py`).

Yaw MAAE (degrees):

| Model | clean | mblur9 | mblur21 | warm | cool | gamma0.6 | gamma1.6 | gray | jpeg30 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| dinov3-320 | 0.25 | 0.26 (+4.1%) | 0.35 (+38.5%) | 0.29 (+14.3%) | 0.29 (+14.3%) | 0.26 (+3.0%) | 0.26 (+2.3%) | 0.28 (+10.0%) | 0.27 (+6.0%) |
| vitt-128 | 0.53 | 0.60 (+14.7%) | 4.75 (+803.8%) | 0.78 (+48.8%) | 0.84 (+60.5%) | 0.61 (+16.1%) | 0.61 (+15.2%) | 0.62 (+17.5%) | 0.61 (+16.1%) |
| vitt-064 | 0.76 | 2.06 (+172.3%) | 26.12 (+3348.2%) | 1.12 (+47.2%) | 1.25 (+64.5%) | 0.89 (+17.1%) | 0.88 (+16.8%) | 0.93 (+22.5%) | 1.14 (+50.2%) |
| hgnetv2-128 | 1.83 | 2.70 (+47.6%) | 10.08 (+450.7%) | 2.57 (+40.4%) | 3.03 (+65.5%) | 2.12 (+15.8%) | 2.03 (+10.8%) | 2.34 (+27.7%) | 2.31 (+25.9%) |
| hgnetv2-064 | 2.34 | 5.73 (+145.4%) | 36.63 (+1468.5%) | 3.11 (+33.4%) | 3.64 (+55.9%) | 2.63 (+12.6%) | 2.57 (+9.9%) | 2.94 (+26.0%) | 3.51 (+50.3%) |
| yawnet-128 | 5.22 | 5.58 (+6.8%) | 14.83 (+183.8%) | 5.33 (+2.1%) | 5.64 (+8.0%) | 5.28 (+1.1%) | 5.29 (+1.2%) | 5.52 (+5.7%) | 5.44 (+4.1%) |
| yawnet-064 | 5.55 | 6.86 (+23.6%) | 47.70 (+759.1%) | 5.72 (+3.0%) | 6.05 (+8.9%) | 5.67 (+2.2%) | 5.63 (+1.5%) | 5.97 (+7.6%) | 6.13 (+10.5%) |

Pitch MAAE (degrees):

| Model | clean | mblur9 | mblur21 | warm | cool | gamma0.6 | gamma1.6 | gray | jpeg30 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| dinov3-320 | 0.21 | 0.22 (+3.6%) | 0.29 (+33.9%) | 0.25 (+18.8%) | 0.26 (+19.8%) | 0.22 (+3.1%) | 0.22 (+2.2%) | 0.24 (+11.6%) | 0.23 (+6.5%) |
| vitt-128 | 0.49 | 0.55 (+13.7%) | 4.38 (+797.4%) | 0.77 (+58.9%) | 0.84 (+71.8%) | 0.56 (+15.6%) | 0.57 (+16.1%) | 0.60 (+22.5%) | 0.56 (+15.3%) |
| vitt-064 | 0.67 | 1.95 (+190.5%) | 21.45 (+3097.6%) | 1.11 (+65.0%) | 1.22 (+81.6%) | 0.79 (+17.8%) | 0.79 (+18.1%) | 0.85 (+26.8%) | 0.99 (+47.2%) |
| hgnetv2-128 | 1.71 | 2.63 (+53.2%) | 9.77 (+470.0%) | 2.77 (+61.7%) | 3.41 (+99.3%) | 2.05 (+19.4%) | 1.94 (+13.0%) | 2.28 (+33.0%) | 2.24 (+31.0%) |
| hgnetv2-064 | 2.02 | 5.18 (+156.2%) | 26.53 (+1212.6%) | 3.22 (+59.2%) | 3.70 (+82.9%) | 2.34 (+15.9%) | 2.23 (+10.5%) | 2.54 (+25.9%) | 3.11 (+53.8%) |
| yawnet-128 | 7.54 | 7.64 (+1.2%) | 14.31 (+89.7%) | 7.75 (+2.8%) | 7.91 (+4.8%) | 7.59 (+0.6%) | 7.55 (+0.1%) | 7.63 (+1.2%) | 7.63 (+1.1%) |
| yawnet-064 | 7.55 | 8.31 (+10.2%) | 36.14 (+378.9%) | 7.80 (+3.4%) | 8.07 (+6.9%) | 7.60 (+0.7%) | 7.61 (+0.8%) | 7.73 (+2.4%) | 7.81 (+3.6%) |

Observations: motion blur is the main weakness; on a 64 px input the 21 px blur spans a third of the crop, hence the extreme rows (as noted above, the degradations are applied in input pixels, so the 64 px rows are not comparable with the 128 / 320 px rows). The color-temperature shifts (warm / cool) cost the ViT-T and PP-HGNetV2 students 35–100 %, while the MBConv student moves by only a few percent; models with small absolute errors naturally show larger relative changes. Gamma, grayscale and JPEG recompression are comparatively mild for every model.

**How to read these numbers (important):** the models are trained with `--unified` (train + validation merged on purpose), so the yaw/pitch values above measure how well each model fits the training distribution and **must not be compared with published benchmark results**. They are selection metrics, useful for comparing the rows against each other. The yaw/pitch columns are the same in both tables because the roll branch does not change the body; the integrated table adds the roll estimate, the derotation and their cost. The roll column is the roll branch — a YawNet w1.0 (0.77M) distilled online from a DINOv3 ViT-L/16 roll teacher trained at 320×320, run at 64×64 for the 64 px students and at 128×128 for hfhpe_dinov3 and the 128 px hfhpe_vitt / hfhpe_hgnetv2 / hfhpe_yawnet — evaluated on **wedge-free canvas rotations** of the validation crops, an honest protocol in which the four corners are filled with real context pixels instead of border replication.

## 2. Features

- **SynthYawPitchPose dataset tooling**: JSONL label format (`data/yawpitchpose/{train,val}.jsonl` + 320px whole-head crops) covering seven synthetic generation sources, full-surround yaw, pitch −120…+120°, and large-accessory occlusion (masks, sunglasses, hats). Includes the audit tooling used to review every label by eye and correct 35.6k of them (mirror fixes, high-pitch frontal audit, per-cell visual sheets).
- **Angle regression that cannot wrap around**: every angle is a biternion (unit `cos/sin`) trained with a von Mises NLL; a κ head per angle outputs the concentration = confidence. No discontinuity at 0/360° and no gimbal handling in the loss.
  <img width="795" height="601" alt="image" src="https://github.com/user-attachments/assets/b10c16e4-df40-4300-81c6-92388a49f39c" />
- **Online teacher → student distillation** (`scripts/distill_yawnet.py`): the teacher (DINOv3 ViT-L/16 @320) is run on the fly on the same crops as the student (KD 0.7 / GT 0.3), with WSD learning-rate schedule, EMA, inverse-frequency yaw balancing and warm-start (`--init-student`) between generations.
- **Roll branch with label-free real-image adaptation** (`scripts/train_hfhpe_roll.py`): roll GT is the exact synthetic rotation θ applied at load time, so any upright crop is a training sample. Real photographs are re-cropped with the same DEIMv2 detector as deployment and mixed in via `--extra-data` — their pose labels are never used as targets. This removed the catastrophic ~180° roll failures on real photos. Supports ViT-T (`--arch vitt`) and DINOv3 ViT-L/16 (`--arch vitl`, 320 px) roll teachers, online KD (`--teacher`; with a higher-resolution teacher the student receives an area-downscaled copy of the same augmented view) and warm-start (`--init-student`). The deployed branch is a YawNet w1.0 distilled from the ViT-L teacher.
- **Wedge-free canvas rotation** (`scripts/build_roll_canvas_synth.py` + the `canvas` field of `roll_dataset.py`): rotating a square crop normally leaves border-replication wedges in the corners whose orientation leaks the applied angle (a shortcut the model learns to read). The canvas build stores a 1.8× context canvas around the identical crop contract; at train time the central contract region is rotated onto the full output while the corners are filled from the real surrounding ring. θ=0 reproduces the deployed crop exactly, so the runtime contract is unchanged.
- **One-pass HFHPE composition** (`scripts/export_hfhpe.py`): roll estimate → GridSample derotation → normalization conversion → yaw/pitch body, fused into a single ONNX graph.
- **κ-gated derotation (optional)** (`--roll-gate sigmoid|relu`): when the roll confidence κ_roll falls below a threshold τ, the in-graph derotation is blended toward the identity rotation, preventing a wrong large-angle derotation from corrupting the yaw/pitch body. Built from Sub/Mul/Sigmoid/Relu/Clip/Sqrt only — no Where/Greater/Cast — for edge-runtime compatibility. The `cos_sin` / `kappa` outputs always stay the raw estimates.
- **Verified ONNX export** (`scripts/export_onnx.py`): onnxslim → onnxsim → graph canonicalization → torch/onnxruntime parity → fixed batch-1 → N-batch conversion with batch 1/2/3 agreement checks → machine audit → I/O-contract metadata injection (readable in Netron).
- **Preview instruments**: 3×3 validation preview sheets (yaw ring + pitch dial) and 3-axis overlay previews for the integrated ONNX model.

## 3. Models and the ONNX I/O contract

| | Teacher dinov3-320 | Student vitt-064 | Student hgnetv2-064 | Student yawnet-064 |
|---|---|---|---|---|
| Backbone | DINOv3 ViT-L/16 (weights not vendored) | ViT-T/16 (own implementation, initialized from `ckpts/vitt_distill.pt`) | PP-HGNetV2-B0, final stage stride-1 (own implementation, initialized from `ckpts/PPHGNetV2_B0_stage1.pth`) | MBConv CNN (own implementation, from scratch) |
| Head | 6 outputs: yaw/pitch biternion + κ each | same | same | same |
| Input normalization | ImageNet mean/std | center05 `x/127.5 − 1` | center05 | center05 |
| Roll branch | YawNet w1.0, 0.77M params, center05, distilled from a DINOv3 ViT-L/16 roll teacher (320×320); run at 128×128 for hfhpe_dinov3 and the 128 px vitt / hgnetv2 / yawnet, and at 64×64 for the 64 px students (the body input is resized to the roll size inside the graph) | | | |

Integrated HFHPE ONNX (`hfhpe_*.onnx`; the yaw/pitch-only exports `*_1x3xSxS.onnx` / `*_kappa_*.onnx` omit the roll fields). The `hfhpe_*_rollgate_*.onnx` variants embed the κ-gated derotation (sigmoid gate; τ=20 for the three students, τ=95 for hfhpe_dinov3 and the 128 px hfhpe_vitt / hfhpe_hgnetv2 / hfhpe_yawnet: with the 128×128 roll branch κ_roll saturates on almost every real image, so any non-saturated value signals an unreliable roll estimate) with the exact same I/O contract; the gate settings are also recorded in the ONNX metadata (`roll_gate`):

```
# RGB, center05 normalization x/127.5 - 1 (also for the dinov3 variant:
# the ImageNet conversion happens inside the graph). S = 64 (students) / 320 (dinov3)
input   images   float32 [N, 3, S, S]
# unit-vector pairs; angle_deg = degrees(atan2(sin, cos))
output  cos_sin  float32 [N, 6]   # [cos_yaw, sin_yaw, cos_pitch, sin_pitch, cos_roll, sin_roll]
# von Mises concentration = confidence per angle (~1e-3 ... 100)
output  kappa    float32 [N, 3]   # [kappa_yaw, kappa_pitch, kappa_roll]
```

Angle convention (also written into every ONNX file as metadata):

- **yaw**: 0° = facing the camera, +90° = turned to the viewer's left, 180° = back of the head, 270° = viewer's right (mod 360).
- **pitch**: positive = looking up, negative = seen from above; range about −120…+120°, values beyond +90° describe the back-tilt band.
- **roll**: positive = image counter-clockwise (top of the head moves toward the viewer's left), full 360°.

Crop contract (identical in training, evaluation and deployment): DEIMv2-Wholebody49 head box (class 7; score ≥ 0.30 when building the training and evaluation crops, ≥ 0.70 by default in the demo) → expand each side by 5% → square of the longer side, clamped at the image border → resize to S×S. Two files are written per export: fixed batch-1 `*_1x3xSxS.onnx` and `*_Nx3xSxS.onnx` with a symbolic batch axis, verified to agree at batch 1/2/3.

## 4. Setup

Prerequisites: Linux, Python 3.13 (pinned in `.python-version`), [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU (torch built for CUDA 12.8; an 8 GB GPU is enough for inference and export, full training targets a 96 GB class GPU). Dependencies are pinned with `==` in `pyproject.toml` and locked in `uv.lock`.

```bash
git clone https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation.git
cd High-Angle_Robust_Fast_Head_Pose_Estimation
uv sync --frozen                                   # default: onnxruntime-gpu 1.22.0 (the validated toolchain)
uv sync --frozen --no-group ort --group tensorrt   # alternative: onnxruntime-gpu 1.26.0 (TensorRT EP, BF16)
# NOTE: a plain `uv run` re-syncs the DEFAULT groups; keep the same group flags on `uv run`
# or use `uv run --no-sync` after the manual sync.
```

### 4.1 Weights (`ckpts/`, `models/` — none of them are bundled)

| File | Purpose | Source / license |
|---|---|---|
| `ckpts/dinov3_vitl16_pretrain_lvd1689m-*.pth` | Teacher backbone | Distributed by Meta's [DINOv3](https://github.com/facebookresearch/dinov3) (DINOv3 License, no redistribution) |
| `ckpts/vitt_distill.pt` | Initial weights of the ViT-T/16 student | ImageNet-pretrained weights distributed with DEIMv2 (Apache-2.0) |
| `ckpts/PPHGNetV2_B0_stage1.pth` | Initial backbone weights of the CNN student | ImageNet-pretrained weights distributed with DEIMv2 (Apache-2.0) |
| `models/deimv2_wholebody49_boxes_only.onnx` | Head detection for dataset building, the crop contract and evaluation | [DEIMv2-Wholebody49](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/488_DEIMv2-Wholebody49) (Apache-2.0) |

### 4.2 Data (`data/`, not bundled)

- `data/yawpitchpose/` — the SynthYawPitchPose dataset: `labels_fixed.jsonl` / `train.jsonl` / `val.jsonl` plus 320px whole-head crops under `images/`. Seven synthetic generation sources plus 1,681 self-portrait crops of the author (`synthetic_real_001` / `synthetic_real_002`, teacher pseudo-labels corrected by hand). Distributed separately under the `PINTO-CC-BY-4.0-License-Version-1.0` (see section 8); not bundled with this repository.
- Roll-adaptation crops (optional, `--extra-data` of the roll training): any directory of upright whole-head crops with a `manifest.jsonl` (`{"image": "images/<name>.jpg", ...}` per line). Real photo collections can be converted with the preprocessing tooling in `scripts/`; their pose labels are never used as training targets.

## 5. Training, evaluation and export

### 5.1 Scripts

```bash
# Teacher (DINOv3 ViT-L/16 @320, yaw+pitch, WSD, EMA; ~7 h / 90 epochs on a 96 GB GPU)
uv run python scripts/train_teacher_dinov3.py \
--vram 96 --variant vitl16 --size 320 \
--pitch-head --unified \
--lr-schedule wsd --epochs 90 --decay-epochs 10 \
--ema-decay 0.999 --grad-clip 1.0 \
--data data/yawpitchpose --tag teacher

# Students (online distillation from the teacher; --init-student warm-starts from a previous run)
## ViT-T @64
uv run python scripts/distill_yawnet.py \
--teacher runs/dinov3_vitl16_320_yp_unified_teacher \
--student-arch vitt --student-size 64 --pitch-head --unified \
--vram 96 --lr-schedule wsd --epochs 75 --decay-epochs 20 --ema-decay 0.999 \
--data data/yawpitchpose --tag v1
## ViT-T @128 (warm-started from the 64 px student; pairs with the 128 px roll branch)
# two cycles: 150/100 from the 64 px student, then the same command again with
# --epochs 300 and --init-student runs/vitt_distill_128_yp_unified_v1 (the first cycle)
uv run python scripts/distill_yawnet.py \
--teacher runs/dinov3_vitl16_320_yp_unified_teacher \
--init-student runs/vitt_distill_064_yp_unified_v9 \
--student-arch vitt --student-size 128 --pitch-head --unified \
--vram 96 --lr 2e-4 --lr-backbone 2e-5 --grad-clip 1.0 \
--lr-schedule wsd --epochs 150 --decay-epochs 100 --ema-decay 0.999 \
--data data/yawpitchpose --tag v1
## PP-HGNetV2-B0 CNN @64
uv run python scripts/distill_yawnet.py \
--teacher runs/dinov3_vitl16_320_yp_unified_teacher \
--student-arch hgnetv2 --student-size 64 --pitch-head --unified \
--vram 96 --lr 2e-4 --lr-backbone 2e-5 --grad-clip 1.0 \
--lr-schedule wsd --epochs 400 --decay-epochs 100 --ema-decay 0.999 \
--data data/yawpitchpose --tag v1
## PP-HGNetV2-B0 CNN @128 (warm-started from the 64 px student; pairs with the 128 px roll branch)
uv run python scripts/distill_yawnet.py \
--teacher runs/dinov3_vitl16_320_yp_unified_teacher \
--init-student runs/hgnetv2_distill_064_yp_unified_v9 \
--student-arch hgnetv2 --student-size 128 --pitch-head --unified \
--vram 96 --lr 2e-4 --lr-backbone 2e-5 --grad-clip 1.0 \
--lr-schedule wsd --epochs 400 --decay-epochs 100 --ema-decay 0.999 \
--data data/yawpitchpose --tag v1
## MBConv CNN @64 (from scratch)
uv run python scripts/distill_yawnet.py \
--teacher runs/dinov3_vitl16_320_yp_unified_teacher \
--student-arch yawnet --student-size 64 --pitch-head --unified \
--vram 96 --lr-schedule wsd --epochs 285 --decay-epochs 20 --ema-decay 0.999 \
--data data/yawpitchpose --tag v1
## MBConv CNN @128 (warm-started from the 64 px student; pairs with the 128 px roll branch)
uv run python scripts/distill_yawnet.py \
--teacher runs/dinov3_vitl16_320_yp_unified_teacher \
--init-student runs/yawnet_distill_064_yp_unified_v9 \
--student-arch yawnet --student-size 128 --pitch-head --unified \
--vram 96 --lr-schedule wsd --epochs 150 --decay-epochs 100 --ema-decay 0.999 \
--data data/yawpitchpose --tag v1

# Roll branch, step 0: build the wedge-free 1.8x canvas dataset from the
# SynthYawPitchPose sources (theta=0 reproduces the deployed crop exactly)
uv run python scripts/build_roll_canvas_synth.py

# Roll branch, step 1: DINOv3 ViT-L/16 roll teacher at 320 px (roll GT = the
# synthetic rotation applied at load time; --extra-data mixes in the
# label-free real-image crops of section 4.2). --vram 8 fits an 8 GB GPU
# (last 8 blocks fine-tuned, micro-batch 8 x accumulation 8)
uv run python scripts/train_hfhpe_roll.py \
--arch vitl --size 320 --lr 2e-4 \
--unified --vram 96 \
--lr-schedule wsd --epochs 90 --decay-epochs 10 \
--ema-decay 0.999 --grad-clip 1.0 \
--data data/roll_canvas_synth \
--extra-data data/real_upright_crops \
--tag canvas

# Roll branch, step 2: deployed student (YawNet w1.0, 0.77M, 64 px) distilled
# online from the ViT-L teacher (alpha*KD + beta*GT). The augmented view is
# rendered once at the teacher resolution and area-downscaled for the student
uv run python scripts/train_hfhpe_roll.py \
--width 1.0 \
--teacher runs/hfhpe_roll_vitl_320_unified_canvas \
--unified --vram 96 \
--lr-schedule wsd --epochs 300 --decay-epochs 100 \
--ema-decay 0.999 --grad-clip 1.0 \
--data data/roll_canvas_synth \
--extra-data data/real_upright_crops \
--tag kd_vitl_canvas_w100

# Roll branch, step 3 (hfhpe_dinov3 only): the same student at 128 px, warm-started
# from the 64 px one; the 320 px body input is downscaled to 128 px inside the graph
uv run python scripts/train_hfhpe_roll.py \
--width 1.0 --size 128 \
--teacher runs/hfhpe_roll_vitl_320_unified_canvas \
--init-student runs/hfhpe_roll_064_unified_kd_vitl_canvas_w100 \
--unified --vram 96 \
--lr-schedule wsd --epochs 200 --decay-epochs 100 \
--ema-decay 0.999 --grad-clip 1.0 \
--data data/roll_canvas_synth \
--extra-data data/real_upright_crops \
--tag kd_vitl_canvas_w100

# ONNX: yaw/pitch body (fixed batch-1 + N-batch, parity/audit/metadata included)
uv run python scripts/export_onnx.py --ckpt runs/vitt_distill_064_yp_unified_v9

uv run python scripts/export_onnx.py --ckpt runs/vitt_distill_064_yp_unified_v9 \
--with-kappa \
--nbatch-atol 1e-3

# ONNX: integrated one-pass HFHPE (roll + derotation + body in one graph)
uv run python scripts/export_hfhpe.py \
--roll-ckpt runs/hfhpe_roll_064_unified_kd_vitl_canvas_w100 \
--body-ckpt runs/vitt_distill_064_yp_unified_v9 \
--kappa-tol 1e-2 \
--nbatch-atol 1e-3

# ONNX: same, with the kappa-gated derotation baked in (stem gains _rollgate)
uv run python scripts/export_hfhpe.py \
--roll-ckpt runs/hfhpe_roll_064_unified_kd_vitl_canvas_w100 \
--body-ckpt runs/vitt_distill_064_yp_unified_v9 \
--roll-gate sigmoid --roll-gate-tau 20 \
--kappa-tol 1e-2 \
--nbatch-atol 1e-3

# hfhpe_dinov3: 320 px body + 128 px roll branch, gate at the κ saturation level
uv run python scripts/export_hfhpe.py \
--roll-ckpt runs/hfhpe_roll_128_unified_kd_vitl_canvas_w100 \
--body-ckpt runs/dinov3_vitl16_320_yp_unified_teacher \
--roll-gate sigmoid --roll-gate-tau 95 \
--kappa-tol 1e-2 \
--nbatch-atol 1e-3
# the 128 px vitt / hgnetv2 / yawnet are exported the same way (--body-ckpt runs/<arch>_distill_128_yp_unified_v9)
# with the 128 px roll branch and --roll-gate-tau 95

# Validation preview sheets (3x3; deterministic per --set)
uv run python scripts/render_preview.py \
--ckpt runs/dinov3_vitl16_320_yp_unified_teacher --data data/yawpitchpose --set 1

# 3-axis overlay preview of the integrated ONNX (X=red right cheek, Y=green chin, Z=blue nose)
uv run python scripts/render_hfhpe_preview.py \
--onnx runs/vitt_distill_064_yp_unified_v9/hfhpe_vitt_1x3x64x64.onnx --set 1

# 3-axis sheet of nine roll-canvas crops rotated by known angles (roll GT = the rotation),
# then the teacher's CLS attention overlaid on the same nine cells
uv run python scripts/render_hfhpe_axes_sheet.py \
--onnx runs/vitt_distill_064_yp_unified/hfhpe_vitt_1x3x64x64.onnx
uv run python scripts/render_hfhpe_axes_heatmap.py \
--sheet-json runs/vitt_distill_064_yp_unified/hfhpe_axes_sheet_roll_canvas_synth.json \
--ckpt runs/dinov3_vitl16_320_yp_unified_teacher
```

Training logs are written to `runs/<run>/train_log.jsonl` (validation metrics per epoch); the best checkpoint is `runs/<run>/best_<val>.pt` and the resume checkpoint is `last.pt` (`--resume`).

Both checkpoints also store the full argument list (dataset roots, teacher / initial-weight paths, hyperparameters) and, for distilled students, the teacher's path and metrics. Before sharing a checkpoint, run `scripts/sanitize_checkpoint.py runs/<run>` to write a copy under `runs/public/<run>/` that keeps only the weights and the keys the loaders need; `export_onnx.py --ckpt runs/public/<run>` then produces identically named ONNX files.

### 5.2 Training-time augmentation

All augmentation is applied on the fly in the datasets (`scripts/yaw_dataset.py`, `scripts/roll_dataset.py`) with the functions of `scripts/augment.py`; validation uses none of it (a plain INTER_LINEAR resize, or a fixed rotation for the roll branch). The training scripts keep the defaults below.

**Yaw / pitch bodies (teacher and distilled students).** All geometric terms are composed into one 3×3 projective matrix and the crop is warped exactly once (INTER_LINEAR, replicated border), following the single-warp design of HRFFA. The input is the 5 % margin square head crop.

| Augmentation | Probability | Range | Label update |
|---|---|---|---|
| Horizontal flip | 0.5 | — | yaw → −yaw (exact), pitch unchanged |
| Scale | always | 0.85–1.10 | none |
| Translation | always | ±5 % of the side (x and y) | none |
| Small in-plane roll | 0.3 | ±8° | none (small-angle approximation) |
| Camera-rotation homography (azimuth ψ and elevation φ together) | 0.3 | ψ ±10°, φ ±12°, focal ratio 1.2 | yaw += 0.166·ψ (measured coefficient), pitch += −0.166·φ (provisional, mirrored from the yaw measurement) |

The degradations below leave the labels unchanged and run in this order after the warp:

| Augmentation | Probability | Details |
|---|---|---|
| Low-resolution jitter | 0.35 | downscale to a random side of 20 px … original, then upscale back (surveillance-camera look) |
| Brightness / contrast / gamma | 0.8 | gain 0.6–1.4, offset ±25, gamma 0.7–1.4 |
| Grayscale | 0.2 | — |
| Gaussian noise | 0.3 | σ 3–12 |
| Gaussian blur | 0.3 | kernel 3 or 5 |
| JPEG recompression | 0.3 | quality 35–84 |
| Motion blur | 0.2 | length 1–6 % of the side, uniform direction |
| Random erasing | 0.25 | 1–2 rectangles of 10–28 % of the side, filled with noise or the mean color (synthetic occlusion) |

During distillation the whole chain is applied once at the teacher resolution (320 px); the student sees an INTER_LINEAR downscale of that same view, so teacher and student differ only in resolution and the label updates are shared.

**Roll branch.** The in-plane rotation θ ~ U[0°, 360°) is the ground truth itself. It is composed with a horizontal flip (p = 0.5, applied before the rotation), a scale of 0.85–1.10 and a translation of ±5 % into a single affine warp; on the 1.8× canvases the central 1/1.8 is cut out after the rotation, so the corners are real pixels rather than replicated borders. The same degradations as above follow (low-resolution jitter 0.35, photometric chain, motion blur 0.2, random erasing 0.25). The camera-rotation homography and the small roll are not used here. When distilling from the ViT-L roll teacher the view is rendered at the teacher resolution and area-downscaled for the student.

The pitch coefficient of the camera elevation (−0.166) is a mirrored assumption from the measured yaw coefficient and is marked as provisional in the code.

## 6. ONNX demo

Camera input (`-v 0`) requests MJPEG (`MJPG`) at 640×480, matching `simpcamcap`,
and prints the negotiated capture format at startup. Some UVC cameras support VGA
only in MJPEG; leaving the format unspecified can select a YUYV mode that produces
`VIDEOIO(V4L2): select() timeout`. A failed camera read is reported as an error;
video files still stop normally at the end of the stream.

Two-stage inference demo (adapted from the [HRFFA](https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment) demo; no face alignment or YawNet stage): DEIMv2-Wholebody49 head detection → integrated HFHPE ONNX on every head crop. It draws the head boxes, a 3-axis pose overlay per head (X = red right cheek, Y = green chin, Z = blue nose), the per-head yaw/pitch/roll text, and a top-right yaw ring for the largest head (0 = facing the camera, 90 = viewer's left).

```bash
# images
uv run python demo/demo_hfhpe_onnx.py \
-i images_dir \
-o output_dir

# video file / camera 0
uv run python demo/demo_hfhpe_onnx.py \
-v input.mp4 \
-o output_dir

uv run python demo/demo_hfhpe_onnx.py \
-v 0 \
-o output_dir

uv run python demo/demo_hfhpe_onnx.py \
-pm models/hfhpe_vitt_1x3x64x64.onnx \
-v 0 \
-o output_dir \
--head_score_threshold 0.35

# CUDA detector (default) + TensorRT pose EP (engine cache under models/trt_cache)
uv run --no-group ort --group tensorrt python demo/demo_hfhpe_onnx.py \
-v 0 \
-o output_dir \
-d tensorrt \
--head_score_threshold 0.35

# select both EPs explicitly
uv run --no-group ort --group tensorrt python demo/demo_hfhpe_onnx.py \
-v 0 \
-o output_dir \
--detector_device cuda \
--pose_device tensorrt

# CPU for both stages
uv run python demo/demo_hfhpe_onnx.py \
-v 0 \
-o output_dir \
--detector_device cpu \
--pose_device cpu

# draw bounding boxes for all non-attribute classes (pose estimation remains head-only)
uv run python demo/demo_hfhpe_onnx.py \
-v 0 \
-o output_dir \
--detect_all_classes \
--head_score_threshold 0.35

# swap the pose model
uv run python demo/demo_hfhpe_onnx.py \
-pm models/hfhpe_hgnetv2_Nx3x64x64.onnx \
-v 0 \
-o output_dir

# N-batch graph of the default model (all heads of a frame in one pass)
uv run python demo/demo_hfhpe_onnx.py \
-pm models/hfhpe_vitt_rollgate_Nx3x64x64.onnx \
-v 0 \
-o output_dir

# without the kappa gate (same I/O; always derotates by the estimated roll)
uv run python demo/demo_hfhpe_onnx.py \
-pm models/hfhpe_vitt_Nx3x64x64.onnx \
-v 0 \
-o output_dir
```

Defaults: detector `models/deimv2_dinov3_s_wholebody49_boxes_only.onnx`, pose `models/hfhpe_vitt_rollgate_1x3x64x64.onnx` (the κ-gated ViT-T model, fixed batch-1 graph: heads are processed one by one; N-batch graphs are also accepted and run all heads of a frame in one pass). The crop contract of section 3 (5% per-side expansion, long-side square clamped at the border, resize to S×S) is applied as-is to every head with score ≥ 0.70 (`--head_score_threshold`). Input normalization is detected per model: the pose model from its ONNX metadata (center05, or ImageNet for the dinov3 yaw/pitch-only graph), the detector from its graph (ImageNet mean/std for DINOv3-based detectors, plain /255 for HGNetV2-based ones; `--detector_norm div255|imagenet` overrides it — the training crops were built with `div255`). Keys on video / camera input: ESC = quit, `b` = head boxes, `a` = 3-axis overlay, `t` = angle text, `r` = yaw ring. `--save_raw_predictions` writes one JSON per image / frame (bbox, score, yaw/pitch/roll, κ); `--yaw_smooth_tau 0.15` smooths the ring display with a circular EMA (display only). For camera input the recording (`output.mp4`) is kept in real time: a webcam in low light extends its exposure and delivers far fewer frames per second than its nominal rate (5 to 15 fps instead of 30 on the test camera), so frames are repeated or skipped by wall-clock time instead of being written one per captured frame, which would play back fast-forwarded; the measured camera frame rate is printed at the end.

The detector graph is pruned at load time to the nodes that feed `label_xyxy_score`. The `*_boxes_only.onnx` files still carry the instance-mask branch of the graphs they were derived from (only the `masks` output was removed); its `Einsum` runs over the top-K rows of class 0 (body) and receives zero-size tensors on frames without any such row, e.g. extremely dark frames, which crashes onnxruntime (CUDA EP: `CUBLAS_STATUS_INVALID_VALUE` at `/postprocessor/Einsum`, CPU EP: segmentation fault). The pruned graph produces identical `label_xyxy_score` values and runs about 5% faster.

Execution providers are selected independently with `--detector_device` (default: `cuda`) and `--pose_device` (default: CUDA if available, otherwise CPU). Both accept `cpu`, `cuda`, `cuda:0` or `tensorrt`. The existing `-d` / `--device` flags are aliases for `--pose_device` and now affect only the pose estimator; use `--detector_device tensorrt --pose_device tensorrt` for TensorRT on both stages, or `--detector_device cpu --pose_device cpu` for CPU on both. TensorRT falls back to CUDA (when available), then CPU; CUDA falls back to CPU for unsupported operations. `--inference_type` and `--trt_cache_dir` apply to whichever stages use TensorRT.

`--detect_all_classes` draws bounding boxes for non-attribute detector classes with a score at or above `--head_score_threshold`, except face (16), eyes (17), nose (18), mouth (19) and ears (20), whose boxes are hidden to keep the head-pose axes clear. Age (1–2), gender (3–4), head direction (8–15) and left/right subclasses (23–24, 27–28, 30–31, 33–34, 37–38, 40–41, 43–44, 46–47) are also omitted because they share their parent class's box in the [Wholebody49 annotation definition](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/488_DEIMv2-Wholebody49#2-annotation). Head (7) boxes remain visible and keep their green color; other classes use a fixed palette of bright colors selected by class ID, with a black outline for visibility on light backgrounds. Pose estimation, angle overlays, the yaw ring and saved pose JSON still use only head detections (class 7). `--disable_bbox` and the `b` key hide/show all bounding boxes in this mode. Without the flag, only head boxes are drawn.

## 7. Repository layout

```
scripts/
  # models
  dinov3_yaw.py vitt_yaw.py hgnetv2_yaw.py pphgnetv2.py yawnet.py hfhpe.py
  # data pipeline
  build_yawpose_dataset.py build_yawpitchpose_dataset.py yaw_dataset.py
  build_roll_canvas_synth.py roll_dataset.py augment.py
  # training / distillation
  train_teacher_dinov3.py distill_yawnet.py train_yawnet.py train_hfhpe_roll.py
  ema.py vram_presets.py
  # export (onnxslim/onnxsim -> parity -> N-batch -> audit -> metadata)
  export_onnx.py export_hfhpe.py add_onnx_metadata.py count_macs_onnx.py
  sanitize_checkpoint.py       distribution copy of a checkpoint without the training-environment details
  # evaluation / previews
  eval_*.py (eval_style_shift.py / eval_pose_bins.py: the style-shift and pose-bin tables)
  val_preview.py render_preview.py render_hfhpe_preview.py
  render_roll_preview.py render_hfhpe_roll_preview.py plot_*.py
  render_hfhpe_axes_sheet.py render_hfhpe_axes_heatmap.py   3-axis sheet of nine rotated crops and the teacher's CLS attention on the same cells
  # label auditing (mirror fixes, screening, verification)
  fix_*.py screen_*.py verify_*.py check_mirror_fix.py relabel_rear_teacher.py
demo/
  demo_hfhpe_onnx.py           two-stage ONNX demo (head detection -> yaw/pitch/roll)
ckpts/ models/ data/ runs/   weights, detector, datasets and training outputs (not tracked by git)
```

## 8. Technical Articles and Datasets

- The dataset is available under the `PINTO-CC-BY-4.0-License-Version-1.0` at the end of the article below. Please note that this is a paid article.
  - The `PINTO-CC-BY-4.0-License-Version-1.0` has substantially the same legal effect as the `CC-BY-4.0`.
     |Yaw,Pitch|Roll|
    |:-:|:-:|
    |<img width="672" height="720" alt="val_preview_best" src="https://github.com/user-attachments/assets/947c747d-1792-4c64-9ac3-b594b9f20aa5" />|<img width="640" height="640" alt="preview_rotcrop" src="https://github.com/user-attachments/assets/39adf5f2-9f06-4cc8-b7a5-bc58a8a106ff" />|
- The weights for the teacher model are distributed under the MIT License in the article below.
- [506_HFHPE - Design of a Lightweight and Fast 6D Head Pose Estimation Model](https://note.com/pinto0309/n/n5be2ff7ec873?sub_rt=share_pw)
- Technical explanation via video (Japanese audio, with subtitles.)

  [![HFHPE Demo](https://github.com/user-attachments/assets/a0262aab-ec68-4a3d-aa54-b53cd96dcc7a)](https://youtu.be/ICwa_ed6vIM)

- Technical explanation via video (English audio)

  [![HFHPE Demo](https://github.com/user-attachments/assets/c1507f17-803c-4fd4-a34b-f1a8cb807b2d)](https://youtu.be/HAuE1da14vk)

## 9. License

- Code: [MIT License](LICENSE) (Copyright (c) 2026 Katsuya Hyodo).
- Weights: DINOv3 (Meta, DINOv3 License), vitt_distill and PP-HGNetV2 (DEIMv2, Apache-2.0) are neither bundled nor redistributed. Check the derived-work terms of these sources before distributing trained HFHPE weights or ONNX files.

## 10. Citation and acknowledgements

### 10.1 Citing this repository

```bibtex
@software{hyodo2026hfhpe,
  author = {Katsuya Hyodo},
  title  = {{HFHPE}: High-Angle Robust Fast Head Pose Estimation},
  url    = {https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation},
  year   = {2026},
  month  = {10},
  doi    = {10.5281/zenodo.23252206},
}
```

### 10.2 Backbones, pretrained weights and tools

**DINOv3** — teacher backbone (ViT-L/16); weights distributed under the DINOv3 License. Paper: https://arxiv.org/abs/2508.10104 · Code: https://github.com/facebookresearch/dinov3

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

**DEIMv2** — source of the ImageNet-pretrained PP-HGNetV2-B0 and ViT-T weights used by the students (Apache-2.0). Paper: https://arxiv.org/abs/2509.20787 · Code: https://github.com/Intellindust-AI-Lab/DEIMv2

```bibtex
@article{huang2025deimv2,
  title   = {Real-Time Object Detection Meets {DINOv3}},
  author  = {Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
  journal = {arXiv preprint arXiv:2509.20787},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.20787}
}
```

**DEIMv2-Wholebody49** — whole-head detection for dataset building, the crop contract and the evaluation harness (Apache-2.0). https://github.com/PINTO0309/PINTO_model_zoo/tree/main/488_DEIMv2-Wholebody49

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

**6DRepNet360** — reference head-pose model used by the dataset audit tooling, not by the deployed graphs (code MIT; weights not bundled). The yaw convention of SynthYawPitchPose (0° = facing the camera, +90° = viewer's left) follows its output convention, and its ONNX export (`models/sixdrepnet360_1x3x224x224_full.onnx`, PINTO_model_zoo #423) was used to check the sign of the synthetic yaw labels (`verify_labels_sixd.py`, `fix_labels.py`) and to measure the camera-yaw augmentation coefficient (`verify_cam_yaw_sign.py`). Paper: https://doi.org/10.1109/TIP.2024.3378180 · Code: https://github.com/thohemp/6DRepNet360 · ONNX: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/423_6DRepNet360

```bibtex
@article{hempel2024toward,
  author  = {Hempel, Thorsten and Abdelrahman, Ahmed A. and Al-Hamadi, Ayoub},
  title   = {Toward Robust and Unconstrained Full Range of Rotation Head Pose Estimation},
  journal = {IEEE Transactions on Image Processing},
  year    = {2024},
  volume  = {33},
  pages   = {2377--2387},
  doi     = {10.1109/TIP.2024.3378180}
}
```

**HRFFA** — the author's whole-head face-alignment project (MIT). HFHPE reuses its single-warp geometric augmentation design (`augment.py`), its export procedure (simplify → parity → N-batch → audit) and its two-stage ONNX demo as the base of `demo/demo_hfhpe_onnx.py`; the HRFFA ViT-L 68-landmark model (`hrffa_vitl_ibug68`) also served as a second reference for the sign of large-yaw labels during dataset auditing (`study_landmark_yaw.py`, `fix_labels.py`). Code: https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment

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

### 10.3 Methods referenced (reimplemented from the papers; no code was copied)

**BiternionNet** — the biternion output representation (unit `cos/sin` regression) and the von Mises loss used for every angle. The architectures are our own. Paper: https://lucasb.eyer.be/academic/biternions/biternions_gcpr15.pdf · Original implementation (MIT): https://github.com/lucasb-eyer/BiternionNet

```bibtex
@inproceedings{beyer2015biternion,
  title     = {Biternion Nets: Continuous Head Pose Regression from Discrete Training Labels},
  author    = {Beyer, Lucas and Hermans, Alexander and Leibe, Bastian},
  booktitle = {German Conference on Pattern Recognition (GCPR)},
  year      = {2015},
  url       = {https://lucasb.eyer.be/academic/biternions/biternions_gcpr15.pdf}
}
```

### 10.4 Acknowledgements

- Meta AI for DINOv3 and the DEIMv2 authors for the ViT-T and PP-HGNetV2 weights (originally from PaddlePaddle's PaddleClas / PaddleDetection).
- The SynthYawPitchPose generation pools were produced with image-generation APIs; the dataset is distributed separately under the `PINTO-CC-BY-4.0-License-Version-1.0` (see section 8).
- The ONNX post-processing know-how (batch-axis preservation, N-batch derivation, parity-first verification) builds on the author's PersonViT export pipeline and the [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) conventions.

