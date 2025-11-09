# 477_PGC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17564899.svg)](https://doi.org/10.5281/zenodo.17564899) ![GitHub License](https://img.shields.io/github/license/pinto0309/pgc) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/pgc)

Ultrafast pointing gesture classification. Classify whether the finger is pointing near the center of the camera lens.

A model that can only detect slow human gestures is completely worthless. A resolution of 32x32 is sufficient for human hand gesture classification. LSTM and 3DCNN are useless because they are not robust to environmental noise.

https://github.com/user-attachments/assets/19268cf9-767c-441e-abc0-c3abd8dba57a

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|S|494 KB|0.9524|0.43 ms|[Download](https://github.com/PINTO0309/PGC/releases/download/onnx/pgc_s_32x32.onnx)|
|C|875 KB|0.9626|0.50 ms|[Download](https://github.com/PINTO0309/PGC/releases/download/onnx/pgc_c_32x32.onnx)|
|M|1.7 MB|0.9714|0.59 ms|[Download](https://github.com/PINTO0309/PGC/releases/download/onnx/pgc_m_32x32.onnx)|
|L|6.4 MB|0.9782|0.78 ms|[Download](https://github.com/PINTO0309/PGC/releases/download/onnx/pgc_l_32x32.onnx)|

## Setup

```bash
git clone https://github.com/PINTO0309/PGC.git && cd PGC
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

## Inference

```bash
uv run python demo_pgc.py \
-pm pgc_l_32x32.onnx \
-v 0 \
-ep cuda \
-dlr

uv run python demo_pgc.py \
-pm pgc_l_32x32.onnx \
-v 0 \
-ep tensorrt \
-dlr
```

## Arch

<img width="300" alt="pgc_s_32x32" src="https://github.com/user-attachments/assets/f6a6efcc-0b05-4cbe-b578-1c72312c1b61" />

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2025pgc,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/PGC},
  month     = {11},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17564899},
  url       = {https://github.com/PINTO0309/pgc},
  abstract  = {Ultrafast pointing gesture classification.},
}
```

## Acknowledgements
- https://gibranbenitez.github.io/IPN_Hand/: CC BY 4.0 License
  ```bibtex
  @inproceedings{bega2020IPNhand,
    title={IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition},
    author={Benitez-Garcia, Gibran and Olivares-Mercado, Jesus and Sanchez-Perez, Gabriel and Yanai, Keiji},
    booktitle={25th International Conference on Pattern Recognition, {ICPR 2020}, Milan, Italy, Jan 10--15, 2021},
    pages={4340--4347},
    year={2021},
    organization={IEEE}
  }
  ```
- https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34: Apache 2.0 License
  ```bibtex
  @software{DEIMv2-Wholebody34,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 28 classes: body, adult, child, male, female, body_with_wheelchair, body_with_crutches, head, front, right-front, right-side, right-back, back, left-back, left-side, left-front, face, eye, nose, mouth, ear, collarbone, shoulder, solar_plexus, elbow, wrist, hand, hand_left, hand_right, abdomen, hip_joint, knee, ankle, foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34},
    year={2025},
    month={10},
    doi={10.5281/zenodo.10229410}
  }
  ```
- https://github.com/PINTO0309/bbalg: MIT License
- https://github.com/PINTO0309/PGC: MIT License
