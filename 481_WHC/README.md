# 481_WHC
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17690769.svg)](https://doi.org/10.5281/zenodo.17690769) ![GitHub License](https://img.shields.io/github/license/pinto0309/WHC) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/whc)

Waving Hand Classification. Ultrafast 1x3x4x32x32 3DConv gesture estimation.

https://github.com/user-attachments/assets/c6b38d56-48b7-4609-bae1-f607c21ba423

https://github.com/user-attachments/assets/7e9f8763-839f-46d2-98b1-320170f8ed10

|Variant|Size|Seq|F1|CPU<br>inference<br>latency|ONNX<br>static seq|ONNX<br>dynamic seq|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|S|1.1 MB|4|0.9821|0.31 ms|[Download](https://github.com/PINTO0309/WHC/releases/download/onnx/whc_seq_3dcnn_4x32x32.onnx)|[Download](https://github.com/PINTO0309/WHC/releases/download/onnx/whc_seq_3dcnn_T4x32x32.onnx)|
|M|1.1 MB|6|0.9916|0.46 ms|[Download](https://github.com/PINTO0309/WHC/releases/download/onnx/whc_seq_3dcnn_6x32x32.onnx)|[Download](https://github.com/PINTO0309/WHC/releases/download/onnx/whc_seq_3dcnn_T6x32x32.onnx)|
|L|1.1 MB|8|0.9940|0.37 ms|[Download](https://github.com/PINTO0309/WHC/releases/download/onnx/whc_seq_3dcnn_8x32x32.onnx)|[Download](https://github.com/PINTO0309/WHC/releases/download/onnx/whc_seq_3dcnn_T8x32x32.onnx)|

## Data sample

|1|2|3|4|
|:-:|:-:|:-:|:-:|
|<img width="32" height="32" alt="image" src="https://github.com/user-attachments/assets/a5ac4472-9ab9-42ce-85f3-baa93cfb2884" />|<img width="32" height="32" alt="image" src="https://github.com/user-attachments/assets/dc9bb1c7-8757-4fe7-823f-a1be1ac3b5b7" />|<img width="32" height="32" alt="image" src="https://github.com/user-attachments/assets/1399a80d-b249-4c0b-8636-4e58a0ba4188" />|<img width="32" height="32" alt="image" src="https://github.com/user-attachments/assets/c3edcc98-a17b-4c4f-93ab-596f521bb27c" />|

## Inference

```bash
uv run python demo_whc.py \
-wm whc_seq_3dcnn_4x32x32.onnx \
-v 0 \
-ep cuda \
-dlr -dnm -dgm -dhm -dhd

uv run python demo_whc.py \
-wm whc_seq_3dcnn_4x32x32.onnx \
-v 0 \
-ep tensorrt \
-dlr -dnm -dgm -dhm -dhd
```

## Arch

<img width="150" alt="whc_seq_3dcnn_4x32x32" src="https://github.com/user-attachments/assets/66c03363-b62c-4868-9c34-f88574e44466" />

## Ultra-lightweight classification model series
1. [VSDLM: Visual-only speech detection driven by lip movements](https://github.com/PINTO0309/VSDLM) - MIT License
2. [OCEC: Open closed eyes classification. Ultra-fast wink and blink estimation model](https://github.com/PINTO0309/OCEC) - MIT License
3. [PGC: Ultrafast pointing gesture classification](https://github.com/PINTO0309/PGC) - MIT License
4. [SC: Ultrafast sitting classification](https://github.com/PINTO0309/SC) - MIT License
5. [PUC: Phone Usage Classifier is a three-class image classification pipeline for understanding how people
interact with smartphones](https://github.com/PINTO0309/PUC) - MIT License
6. [HSC: Happy smile classifier](https://github.com/PINTO0309/HSC) - MIT License
7. [WHC: Waving Hand Classification](https://github.com/PINTO0309/WHC) - MIT License

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2025whc,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/WHC},
  month     = {11},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17690769},
  url       = {https://github.com/PINTO0309/whc},
  abstract  = {Waving Hand Classification.},
}
```

## Acknowledgments

- https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34: Apache 2.0 License
  ```bibtex
  @software{DEIMv2-Wholebody34,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 28 classes: body, adult, child, male, female, body_with_wheelchair, body_with_crutches, head, front, right-front, right-side, right-back, back, left-back, left-side, left-front, face, eye, nose, mouth, ear, collarbone, shoulder, solar_plexus, elbow, wrist, hand, hand_left, hand_right, abdomen, hip_joint, knee, ankle, foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34},
    year={2025},
    month={10},
    doi={10.5281/zenodo.17625710}
  }
  ```
