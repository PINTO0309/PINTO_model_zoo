# 491_SGC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20824676.svg)](https://doi.org/10.5281/zenodo.20824676) ![GitHub License](https://img.shields.io/github/license/pinto0309/SGC) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/sgc)

Classification of wearing vs. not wearing sunglasses. 48x48.

https://github.com/user-attachments/assets/52b8bce7-3bdb-4ee9-a410-4e0e1322116b

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|P|115 KB|0.8644|0.23 ms|[Download](https://github.com/PINTO0309/SGC/releases/download/onnx/sgc_p_48x48.onnx)|
|N|176 KB|0.9013|0.41 ms|[Download](https://github.com/PINTO0309/SGC/releases/download/onnx/sgc_n_48x48.onnx)|
|T|280 KB|0.9252|0.52 ms|[Download](https://github.com/PINTO0309/SGC/releases/download/onnx/sgc_t_48x48.onnx)|
|S|495 KB|0.9423|0.64 ms|[Download](https://github.com/PINTO0309/SGC/releases/download/onnx/sgc_s_48x48.onnx)|
|L|6.4 MB|0.9554|1.03 ms|[Download](https://github.com/PINTO0309/SGC/releases/download/onnx/sgc_l_48x48.onnx)|

<img width="600" alt="dataset_class_ratio" src="https://github.com/user-attachments/assets/57aff5b6-d12a-433d-a489-d2edc349cffc" />

## Demo

The demo script needs a YOLO whole-body detector ONNX/TFLite model and an SGC sunglasses classifier ONNX model.
Place the detector model in the repository root, or pass its path with `--model`.
Use the ONNX file exported by training for `--sgc_model`.

```bash
python demo_sgc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--sgc_model sgc_is_l_48x48.onnx \
--images_dir path/to/images \
--execution_provider cpu \
--disable_waitKey
```

For a video file:

```bash
python demo_sgc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--sgc_model sgc_is_l_48x48.onnx \
--video path/to/video.mp4 \
--execution_provider cpu
```

For a camera:

```bash
python demo_sgc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--sgc_model sgc_is_l_48x48.onnx \
--video 0 \
--execution_provider cpu \
--disable_generation_identification_mode \
--disable_gender_identification_mode \
--disable_left_and_right_hand_identification_mode \
--disable_headpose_identification_mode
```
```bash
python demo_sgc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--sgc_model sgc_is_l_48x48.onnx \
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
<img width="300" alt="sgc_is_p_48x48" src="https://github.com/user-attachments/assets/7c6b271d-725d-4a0c-97f7-e47ba1580ac5" />

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

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2026sgc,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/SGC},
  month     = {06},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20824676},
  url       = {https://github.com/PINTO0309/sgc},
  abstract  = {Classification of wearing vs. not wearing sunglasses. 48x48.},
}
```