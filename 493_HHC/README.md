# 493_HHC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20931298.svg)](https://doi.org/10.5281/zenodo.20931298) ![GitHub License](https://img.shields.io/github/license/pinto0309/HHC) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/hhc)

Head Hat Classification. HHC is a binary classifier for cropped head images. 48x48.

https://github.com/user-attachments/assets/d719d64e-8c5a-454c-9dc8-3c02b776a1a9

## Classes

| class_id | label |
| --- | --- |
| 0 | `no_wearing_hat` |
| 1 | `wearing_hat` |

Default input size is `48x48`.

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|P|115 KB|0.9150|0.23 ms|[Download](https://github.com/PINTO0309/HHC/releases/download/onnx/hhc_is_p_48x48.onnx)|
|N|176 KB|0.9314|0.41 ms|[Download](https://github.com/PINTO0309/HHC/releases/download/onnx/hhc_is_n_48x48.onnx)|
|T|280 KB|0.9463|0.52 ms|[Download](https://github.com/PINTO0309/HHC/releases/download/onnx/hhc_is_t_48x48.onnx)|
|S|495 KB|0.9702|0.64 ms|[Download](https://github.com/PINTO0309/HHC/releases/download/onnx/hhc_is_s_48x48.onnx)|
|L|6.4 MB|0.9650|1.03 ms|[Download](https://github.com/PINTO0309/HHC/releases/download/onnx/hhc_is_l_48x48.onnx)|

<img width="600" alt="dataset_class_ratio" src="https://github.com/user-attachments/assets/4e2ea89a-9d1a-43a2-b6f4-ba58d63cb31c" />

## Data sample

|1|2|3|4|5|6|
|:-:|:-:|:-:|:-:|:-:|:-:|
|<img width="48" height="48" alt="head_f000060_t002 000_d00_s0 921" src="https://github.com/user-attachments/assets/b4a1e898-d2e1-49d5-9da5-10560f1ce0ca" />|<img width="48" height="48" alt="head_f000030_t001 000_d00_s0 929" src="https://github.com/user-attachments/assets/6ad95d39-24d2-4f8e-9c01-3863223eddfc" />|<img width="48" height="48" alt="head_f000000_t000 000_d00_s0 922" src="https://github.com/user-attachments/assets/e7bd93bb-6e9e-48b9-b814-ce1f6bbd2682" />|<img width="48" height="48" alt="wearing_hat_101005" src="https://github.com/user-attachments/assets/094ae745-fc9b-403c-b674-62a99ae768f0" />|<img width="48" height="48" alt="wearing_hat_100098" src="https://github.com/user-attachments/assets/07bf4469-11bf-48b6-b75e-3f22cfb46138" />|<img width="48" height="48" alt="wearing_hat_100017" src="https://github.com/user-attachments/assets/72fc26f3-a2ca-4233-9709-bbab25c1aad7" />|

## Demo

The demo script needs a YOLO whole-body detector ONNX/TFLite model and an HHC hat classifier ONNX model.
Place the detector model in the repository root, or pass its path with `--model`.
Use the ONNX file exported by training for `--hhc_model`.

```bash
python demo_hhc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--hhc_model hhc_is_l_48x48.onnx \
--images_dir path/to/images \
--execution_provider cpu \
--disable_waitKey
```

For a video file:

```bash
python demo_hhc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--hhc_model hhc_is_l_48x48.onnx \
--video path/to/video.mp4 \
--execution_provider cpu
```

For a camera:

```bash
python demo_hhc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--hhc_model hhc_is_l_48x48.onnx \
--video 0 \
--execution_provider cpu \
--disable_generation_identification_mode \
--disable_gender_identification_mode \
--disable_left_and_right_hand_identification_mode \
--disable_headpose_identification_mode
```
```bash
python demo_hhc.py \
--model yolomit_t_wholebody28_1x3x480x640.onnx \
--hhc_model hhc_is_l_48x48.onnx \
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

## Train

```bash
SIZE=48x48
VAR=p
python -m hhc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/hhc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 1 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=n
python -m hhc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/hhc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 2 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=t
python -m hhc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/hhc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 3 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=s
python -m hhc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/hhc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 4 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```
```bash
SIZE=48x48
VAR=l
python -m hhc train \
--data_root data/dataset.parquet \
--seed 42 \
--output_dir runs/hhc_is_${VAR}_${SIZE} \
--epochs 100 \
--batch_size 256 \
--train_resampling balanced \
--image_size ${SIZE} \
--base_channels 32 \
--num_blocks 8 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--device auto \
--use_amp
```

## Export ONNX

Training exports the best checkpoint to ONNX automatically.
To export a checkpoint manually, run:

```bash
python -m hhc exportonnx \
--checkpoint runs/hhc_is_l_48x48/hhc_best_epoch0067_f1_0.9430.pt \
--output hhc_is_l_48x48.onnx \
--opset 17 \
--device cpu
```

Use the `hhc_best_*.pt` checkpoint from the target run directory.

## Arch
<img width="300" alt="hhc_is_p_48x48" src="https://github.com/user-attachments/assets/07dea100-f9b9-40a8-9cbb-00a66b00c7b7" />

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

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2026hhc,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/HHC},
  month     = {06},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20931298},
  url       = {https://github.com/PINTO0309/hhc},
  abstract  = {Head Hat Classification. HHC is a binary classifier for cropped head images. 48x48.},
}
```
