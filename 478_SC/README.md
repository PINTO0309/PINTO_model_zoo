# 478_SC
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17625710.svg)](https://doi.org/10.5281/zenodo.17625710) ![GitHub License](https://img.shields.io/github/license/pinto0309/SC) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/sc)

Ultrafast sitting classification. 32x24 pixels is sufficient for estimating the state of the whole human body.

https://github.com/user-attachments/assets/635773d8-3826-45fd-ac33-e51fe3695176

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|P|115 KB|0.8923|0.13 ms|[Download](https://github.com/PINTO0309/SC/releases/download/onnx/sc_p_32x24.onnx)|
|N|176 KB|0.9076|0.24 ms|[Download](https://github.com/PINTO0309/SC/releases/download/onnx/sc_n_32x24.onnx)|
|T|279 KB|0.8935|0.31 ms|[Download](https://github.com/PINTO0309/SC/releases/download/onnx/sc_t_32x24.onnx)|
|S|494 KB|0.9168|0.39 ms|[Download](https://github.com/PINTO0309/SC/releases/download/onnx/sc_s_32x24.onnx)|
|C|875 KB|0.9265|0.47 ms|[Download](https://github.com/PINTO0309/SC/releases/download/onnx/sc_c_32x24.onnx)|

## Setup

```bash
git clone https://github.com/PINTO0309/SC.git && cd SC
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

## Inference

```bash
uv run python demo_sc.py \
-pm sc_c_32x24.onnx \
-v 0 \
-ep cuda \
-dlr -dnm -dgm -dhm -dhd

uv run python demo_sc.py \
-pm pgc_c_32x24.onnx \
-v 0 \
-ep tensorrt \
-dlr -dnm -dgm -dhm -dhd
```

## Arch

<img width="300" alt="sc_p_32x24" src="https://github.com/user-attachments/assets/1b0d74b7-ceca-49ae-832d-9ffff80f6945" />

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2025sc,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/SC},
  month     = {11},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17625710},
  url       = {https://github.com/PINTO0309/sc},
  abstract  = {Ultrafast sitting classification.},
}
```

## Acknowledgements
- AVA Actions Download (v2.2) - CC BY 4.0 License
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
- https://github.com/PINTO0309/bbalg: MIT License
- https://github.com/PINTO0309/SC
