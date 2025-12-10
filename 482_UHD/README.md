# 482_UHD
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17790207.svg)](https://doi.org/10.5281/zenodo.17790207) ![GitHub License](https://img.shields.io/github/license/pinto0309/uhd) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/uhd)

Ultra-lightweight human detection. The number of parameters does not correlate to inference speed. For limited use cases, an input image resolution of 64x64 is sufficient. High-level object detection architectures such as YOLO are overkill.

This model is an experimental implementation and is not suitable for real-time inference using a USB camera, etc.

https://github.com/user-attachments/assets/6115de34-ec8a-4649-9e1a-7da46e6f370d

- Variant-S / `w ESE + IoU-aware + ReLU`

|Input<br>64x64|Output|Input<br>64x64|Output|
|:-:|:-:|:-:|:-:|
|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/59b941a4-86eb-41b9-9c0a-c367e6718b4c" />|<img width="640" height="427" alt="00_dist_000000075375" src="https://github.com/user-attachments/assets/5cb2e332-86eb-4933-bf80-3f500173306f" />|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/21791bff-2cf4-4a63-87b5-f6da52bb4964" />|<img width="480" height="360" alt="01_000000073639" src="https://github.com/user-attachments/assets/ed0f9b67-7a05-4e31-8f43-8e82a468ad38" />|
|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/6e5f1203-8aad-4d2a-8439-27a5c8d0a2a7" />|<img width="480" height="360" alt="02_000000051704" src="https://github.com/user-attachments/assets/20fba8fc-5bf6-4485-9165-ce06a1504644" />|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/5e7cd0e1-8840-4a4e-b932-b68dc71f3721" />|<img width="480" height="360" alt="07_000000016314" src="https://github.com/user-attachments/assets/b9aced38-c4eb-4ee7-a6c5-294090748eca" />|
|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/d4b5fc1c-324b-4ed3-9ab7-35c1d022b2d0" />|<img width="480" height="360" alt="28_000000044437" src="https://github.com/user-attachments/assets/499469d3-41a3-406b-a075-8b055297bf6e" />|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/9be24b81-0c6e-48c7-bda1-1372b6ae391f" />|<img width="480" height="360" alt="42_000000006864" src="https://github.com/user-attachments/assets/fed81463-3cd6-4d69-8fc3-5bf114d92ab8" />|
|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/05280dce-4faf-46e1-817c-98e4df4a9d4c" />|<img width="383" height="640" alt="09_dist_000000074177" src="https://github.com/user-attachments/assets/628540af-6407-4e93-b41e-9057671f49ca" />|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/23d932d4-ce48-421e-9d1f-0dd8386852a3" />|<img width="500" height="500" alt="08_dist_000000039322" src="https://github.com/user-attachments/assets/64c744c4-4425-46da-a53b-ffab2fd52060" />|

## Download all ONNX files at once

```bash
sudo apt update && sudo apt install -y gh
gh release download onnx -R PINTO0309/UHD
```

## Models

- Legacy models

  <details><summary>Click to expand</summary>

  - w/o ESE

    |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |N|1.38 M|0.18 G|0.40343|0.93 ms|5.6 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_quality_nopost.onnx)|
    |T|3.10 M|0.41 G|0.44529|1.50 ms|12.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_quality_nopost.onnx)|
    |S|5.43 M|0.71 G|0.44945|2.23 ms|21.8 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_quality_nopost.onnx)|
    |C|8.46 M|1.11 G|0.45005|2.66 ms|33.9 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_quality_nopost.onnx)|
    |M|12.15 M|1.60 G|0.44875|4.07 ms|48.7 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_quality_nopost.onnx)|
    |L|21.54 M|2.83 G|0.44686|6.23 ms|86.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_quality_nopost.onnx)|

  - w ESE

    |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |N|1.45 M|0.18 G|0.41018|1.05 ms|5.8 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_se_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_se_64x64_quality_nopost.onnx)|
    |T|3.22 M|0.41 G|0.44130|1.27 ms|12.9 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_se_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_se_64x64_quality_nopost.onnx)|
    |S|5.69 M|0.71 G|0.46612|2.10 ms|22.8 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_se_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_se_64x64_quality_nopost.onnx)|
    |C|8.87 M|1.11 G|0.45095|2.86 ms|35.5 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_se_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_se_64x64_quality_nopost.onnx)|
    |M|12.74 M|1.60 G|0.46502|3.95 ms|51.0 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_se_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_se_64x64_quality_nopost.onnx)|
    |L|22.59 M|2.83 G|0.45787|6.52 ms|90.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_se_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_se_64x64_quality_nopost.onnx)|

  - ESE + IoU-aware + Swish

    |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |N|1.60 M|0.20 G|0.42806|1.25 ms|6.5 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_se_iou_64x64_quality_nopost.onnx)|
    |T|3.56 M|0.45 G|0.46502|1.82 ms|14.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_se_iou_64x64_quality_nopost.onnx)|
    |S|6.30 M|0.79 G|0.47473|2.78 ms|25.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_se_iou_64x64_quality_nopost.onnx)|
    |C|9.81 M|1.23 G|0.46235|3.58 ms|39.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_se_iou_64x64_quality_nopost.onnx)|
    |M|14.09 M|1.77 G|0.46562|5.05 ms|56.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_se_iou_64x64_quality_nopost.onnx)|
    |L|24.98 M|3.13 G|0.47774|7.46 ms|100 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_se_iou_64x64_quality_nopost.onnx)|

  - ESE + IoU-aware + ReLU

    |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |N|1.60 M|0.20 G|0.40910|0.63 ms|6.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_quality_relu_nopost.onnx)|
    |T|3.56 M|0.45 G|0.44618|1.08 ms|14.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_quality_relu_nopost.onnx)|
    |S|6.30 M|0.79 G|0.45776|1.71 ms|25.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_quality_relu_nopost.onnx)|
    |C|9.81 M|1.23 G|0.45385|2.51 ms|39.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_quality_relu_nopost.onnx)|
    |M|14.09 M|1.77 G|0.47468|3.54 ms|56.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_quality_relu_nopost.onnx)|
    |L|24.98 M|3.13 G|0.46965|6.14 ms|100 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_quality_relu_nopost.onnx)|

  - ESE + IoU-aware + large-object-branch + ReLU

    |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |N|1.98 M|0.22 G|0.40903|0.77 ms|8.0 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_loese.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_loese_nopost.onnx)|
    |T|4.40 M|0.49 G|0.46170|1.40 ms|17.7 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_loese.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_loese_nopost.onnx)|
    |S|7.79 M|0.87 G|0.45860|2.30 ms|31.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_loese.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_loese_nopost.onnx)|
    |C|12.13 M|1.35 G|0.47518|2.83 ms|48.6 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_loese.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_loese_nopost.onnx)|
    |M|17.44 M|1.94 G|0.45816|4.37 ms|69.8 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_loese.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_loese_nopost.onnx)|
    |L|30.92 M|3.44 G|0.48243|7.40 ms|123.7 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_loese.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_loese_nopost.onnx)|

  </details>

- **[For long distances and extremely small objects]** ESE + IoU-aware + ReLU + Distillation

  |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
  |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
  |N|1.60 M|0.20 G|0.55224|0.63 ms|6.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_distill_nopost.onnx)|
  |T|3.56 M|0.45 G|0.56040|1.08 ms|14.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_distill_nopost.onnx)|
  |S|6.30 M|0.79 G|0.57361|1.71 ms|25.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_distill_nopost.onnx)|
  |C|9.81 M|1.23 G|0.56183|2.51 ms|39.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_distill_nopost.onnx)|
  |M|14.09 M|1.77 G|0.57666|3.54 ms|56.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_distill_nopost.onnx)|

- **[For short/medium distance]** ESE + IoU-aware + large-object-branch + ReLU + Distillation

  |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
  |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
  |N|1.98 M|0.22 G|0.54883|0.70 ms|8.0 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_loese_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_loese_distill_nopost.onnx)|
  |T|4.40 M|0.49 G|0.55663|1.18 ms|17.7 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_loese_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_loese_distill_nopost.onnx)|
  |S|7.79 M|0.87 G|0.57397|1.97 ms|31.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_loese_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_loese_distill_nopost.onnx)|
  |C|12.13 M|1.35 G|0.56768|2.74 ms|48.6 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_loese_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_loese_distill_nopost.onnx)|
  |M|17.44 M|1.94 G|0.57815|3.57 ms|69.8 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_loese_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_loese_distill_nopost.onnx)|

## Inference

```bash
usage: demo_uhd.py
[-h]
(--images IMAGES | --camera CAMERA)
--onnx ONNX
[--output OUTPUT]
[--img-size IMG_SIZE]
[--conf-thresh CONF_THRESH]
[--record RECORD]
[--actual-size]

UltraTinyOD ONNX demo (CPU).

options:
  -h, --help
   show this help message and exit
  --images IMAGES
   Directory with images to run batch inference.
  --camera CAMERA
   USB camera id for realtime inference.
  --onnx ONNX
   Path to ONNX model (CPU).
  --output OUTPUT
   Output directory for image mode.
  --img-size IMG_SIZE
   Input size HxW, e.g., 64x64.
  --conf-thresh CONF_THRESH
   Confidence threshold. Default: 0.90
  --record RECORD
   MP4 path for automatic recording when --camera is used.
  --actual-size
   Display and recording use the model input resolution instead of
   the original frame size.
```

- ONNX with post-processing
  ```bash
  uv run demo_uhd.py \
  --onnx ultratinyod_res_anc8_w192_64x64_loese_distill.onnx \
  --camera 0 \
  --conf-thresh 0.90
  ```
- ONNX without post-processing
  ```bash
  uv run demo_uhd.py \
  --onnx ultratinyod_res_anc8_w192_64x64_loese_distill_nopost.onnx \
  --camera 0 \
  --conf-thresh 0.90
  ```

## Arch

|ONNX|LiteRT(TFLite)|
|:-:|:-:|
|<img width="350" alt="ultratinyod_res_anc8_w64_64x64_loese_distill" src="https://github.com/user-attachments/assets/ae5d3c70-8c5e-41f0-ad79-98f4024519a0" />|<img width="350" alt="ultratinyod_res_anc8_w64_64x64_loese_distill_float32" src="https://github.com/user-attachments/assets/c1e909b8-b029-4f0a-acf9-0f8259763ec3" />|

## Ultra-lightweight classification model series
1. [VSDLM: Visual-only speech detection driven by lip movements](https://github.com/PINTO0309/VSDLM) - MIT License
2. [OCEC: Open closed eyes classification. Ultra-fast wink and blink estimation model](https://github.com/PINTO0309/OCEC) - MIT License
3. [PGC: Ultrafast pointing gesture classification](https://github.com/PINTO0309/PGC) - MIT License
4. [SC: Ultrafast sitting classification](https://github.com/PINTO0309/SC) - MIT License
5. [PUC: Phone Usage Classifier is a three-class image classification pipeline for understanding how people
interact with smartphones](https://github.com/PINTO0309/PUC) - MIT License
6. [HSC: Happy smile classifier](https://github.com/PINTO0309/HSC) - MIT License
7. [WHC: Waving Hand Classification](https://github.com/PINTO0309/WHC) - MIT License
8. [UHD: Ultra-lightweight human detection](https://github.com/PINTO0309/UHD) - MIT License

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2025uhd,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/UHD},
  month     = {12},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17790207},
  url       = {https://github.com/PINTO0309/uhd},
  abstract  = {Ultra-lightweight human detection. The number of parameters does not correlate to inference speed.},
}
```
