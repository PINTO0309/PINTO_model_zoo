# 476_OCEC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17505461.svg)](https://doi.org/10.5281/zenodo.17505461) ![GitHub License](https://img.shields.io/github/license/pinto0309/ocec) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/ocec)

Open closed eyes classification. Ultra-fast wink and blink estimation model.

In the real world, attempting to detect eyes larger than 20 pixels high and 40 pixels wide is a waste of computational resources.

https://github.com/user-attachments/assets/2ae9467f-a67f-447e-8704-d16efacdacf1

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|P|112 KB|0.9924|0.16 ms|[Download](https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_p.onnx)|
|N|176 KB|0.9933|0.25 ms|[Download](https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_n.onnx)|
|S|494 KB|0.9943|0.41 ms|[Download](https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_s.onnx)|
|C|875 KB|0.9947|0.49 ms|[Download](https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_c.onnx)|
|M|1.7 MB|0.9949|0.57 ms|[Download](https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_m.onnx)|
|L|6.4 MB|0.9954|0.80 ms|[Download](https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_l.onnx)|

## Arch

<img width="300" alt="ocec_p" src="https://github.com/user-attachments/assets/fa54cf38-0fd4-487a-bf9e-dfbc5401a389" />

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2025ocec,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/OCEC},
  month     = {10},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17505461},
  url       = {https://github.com/PINTO0309/ocec},
  abstract  = {Open closed eyes classification. Ultra-fast wink/blink estimation model.},
}
```

## Acknowledgements
- https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes: Open Data Commons Attribution License (ODC-By) v1.0
  ```bibtex
  @misc{open_closed_eyes2024,
    author = {Michał Młodawski},
    title = {Open and Closed Eyes Dataset},
    month = July,
    year = 2024,
    url = {https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes},
  }
  ```
- https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34 - Apache 2.0
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
- https://github.com/PINTO0309/OCEC - MIT License
  ```bibtex
  @software{hyodo2025ocec,
    author    = {Katsuya Hyodo},
    title     = {PINTO0309/OCEC},
    month     = {10},
    year      = {2025},
    publisher = {Zenodo},
    doi       = {10.5281/zenodo.17505461},
    url       = {https://github.com/PINTO0309/ocec},
    abstract  = {Open closed eyes classification. Ultra-fast wink/blink estimation model.},
  }
  ```
