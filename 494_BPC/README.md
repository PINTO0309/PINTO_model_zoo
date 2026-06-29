# 494_BPC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21022946.svg)](https://doi.org/10.5281/zenodo.21022946) ![GitHub License](https://img.shields.io/github/license/PINTO0309/BPC) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/bpc)

Binary classification of whether the background is simple or complex. 48x48.

https://github.com/user-attachments/assets/5980c05f-be9d-402a-b927-3c60fbd29107

## Classes

| class_id | label |
| --- | --- |
| 0 | `not_plain` |
| 1 | `plain` |

Default input size is `48x48`.

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|P|115 KB|0.9980|0.23 ms|[Download](https://github.com/PINTO0309/BPC/releases/download/onnx/bpc_is_p_48x48.onnx)|
|N|176 KB|0.9983|0.41 ms|[Download](https://github.com/PINTO0309/BPC/releases/download/onnx/bpc_is_n_48x48.onnx)|
|T|280 KB|0.9984|0.52 ms|[Download](https://github.com/PINTO0309/BPC/releases/download/onnx/bpc_is_t_48x48.onnx)|
|S|495 KB|0.9998|0.64 ms|[Download](https://github.com/PINTO0309/BPC/releases/download/onnx/bpc_is_s_48x48.onnx)|
|L|6.4 MB|0.9990|1.03 ms|[Download](https://github.com/PINTO0309/BPC/releases/download/onnx/bpc_is_l_48x48.onnx)|

## Data sample

|1|2|3|4|5|6|
|:-:|:-:|:-:|:-:|:-:|:-:|
|<img width="48" height="48" alt="plain_000001" src="https://github.com/user-attachments/assets/d73b6bac-4aa5-4352-9f03-093b4f911f14" />|<img width="48" height="48" alt="plain_000005" src="https://github.com/user-attachments/assets/dda398cf-01b9-4d23-9dd0-315136a0d9f1" />|<img width="48" height="48" alt="plain_000008" src="https://github.com/user-attachments/assets/5b040a08-823d-4b7c-9f80-f1503d307094" />|<img width="48" height="48" alt="not_plain_000004" src="https://github.com/user-attachments/assets/a6027eeb-288e-487b-af6e-9fc6029c0955" />|<img width="48" height="48" alt="wearing_hat_100021" src="https://github.com/user-attachments/assets/f55c0c03-1604-4b19-86f9-7d9e5e961160" />|<img width="48" height="48" alt="not_plain_000000" src="https://github.com/user-attachments/assets/e440886b-f972-4e3b-845c-dbd2db09c225" />|

<img width="600" alt="dataset_class_ratio" src="https://github.com/user-attachments/assets/9395934a-6920-4e14-9dc5-037ca36d9109" />

## Demo

The demo script needs a YOLO whole-body detector ONNX/TFLite model and an bpc hat classifier ONNX model.
Place the detector model in the repository root, or pass its path with `--model`.
Use the ONNX file exported by training for `--bpc_model`.

```bash
python demo_bpc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--bpc_model bpc_is_s_48x48.onnx \
--images_dir path/to/images \
--execution_provider cpu \
--disable_waitKey
```

For a video file:

```bash
python demo_bpc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--bpc_model bpc_is_s_48x48.onnx \
--video path/to/video.mp4 \
--execution_provider cpu
```

For a camera:

```bash
python demo_bpc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--bpc_model bpc_is_s_48x48.onnx \
--video 0 \
--execution_provider cpu \
--disable_generation_identification_mode \
--disable_gender_identification_mode \
--disable_left_and_right_hand_identification_mode \
--disable_headpose_identification_mode
```
```bash
python demo_bpc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--bpc_model bpc_is_s_48x48.onnx \
--video 0 \
--execution_provider cuda \
--disable_generation_identification_mode \
--disable_gender_identification_mode \
--disable_left_and_right_hand_identification_mode \
--disable_headpose_identification_mode
```

Processed still images are saved under `output/`.
Video input is also recorded to `output.mp4` by default; add `--disable_video_writer` to skip recording.
Use `--execution_provider cuda` or `--execution_provider tensorrt` when the required ONNXRuntime GPU/TensorRT environment is available.

## Arch
<img width="300" alt="bpc_is_p_48x48" src="https://github.com/user-attachments/assets/81aa6a7b-099a-4b55-9f26-e5a69bd5cd01" />

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
9. [MWC: Mask wearing classifier.](https://github.com/PINTO0309/MWC) - MIT License
10. [SGC: Classification of wearing vs. not wearing sunglasses. 48x48.](https://github.com/PINTO0309/SGC) - MIT License
11. [HHC: Head Hat Classification. HHC is a binary classifier for cropped head images. 48x48.](https://github.com/PINTO0309/HHC) - MIT License
12. [BPC: Binary classification of whether the background is simple or complex. 48x48.](https://github.com/PINTO0309/BPC) - MIT License

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2026bpc,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/BPC},
  month     = {06},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21022946},
  url       = {https://github.com/PINTO0309/bpc},
  abstract  = {Binary classification of whether the background is simple or complex. 48x48.},
}
```
