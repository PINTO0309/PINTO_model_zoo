# 479_PUC
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17666420.svg)](https://doi.org/10.5281/zenodo.17666420) ![GitHub License](https://img.shields.io/github/license/pinto0309/PUC)
 [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/puc)

Phone Usage Classifier (PUC) is a three-class image classification pipeline for understanding how people interact with smartphones.

- `classid=0` (`no_action`): No interaction with a smartphone.
- `classid=1` (`point_somewhere`): Pointing the smartphone somewhere other than the camera.
- `classid=2` (`point`): Pointing the smartphone towards the camera.

  https://github.com/user-attachments/assets/56cb5147-78e7-4a9e-85ac-59db79b442c3

  |Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
  |:-:|:-:|:-:|:-:|:-:|
  |P|115 KB|0.9160| 0.24 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_p_48x48.onnx)|
  |N|176 KB|0.9337| 0.39 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_n_48x48.onnx)|
  |T|280 KB|0.9468| 0.51 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_t_48x48.onnx)|
  |S|495 KB|0.9672| 0.66 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_s_48x48.onnx)|
  |C|876 KB|0.9722| 0.73 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_c_48x48.onnx)|
  |M|1.7 MB|0.9774| 0.86 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_m_48x48.onnx)|
  |L|6.4 MB|0.9944| 1.07 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_l_48x48.onnx)|

## Data sample

|no<br>action|no<br>action|point<br>somewhere|point<br>somewhere|point|point|
|:-:|:-:|:-:|:-:|:-:|:-:|
<img width="48" height="48" alt="no_action_008364" src="https://github.com/user-attachments/assets/327c71a0-c636-4ea9-8700-ed5a28a0050e" />|<img width="48" height="48" alt="no_action_008001" src="https://github.com/user-attachments/assets/9d080aa1-fc47-4c83-85a6-9e1f1fa087b8" />|<img width="48" height="48" alt="point_somewhere_002145" src="https://github.com/user-attachments/assets/2d816974-3ddd-4d2c-ae61-0df6cbe5c14f" />|<img width="48" height="48" alt="point_somewhere_002068" src="https://github.com/user-attachments/assets/108d9652-0c07-47f0-8b34-428ee5f23dfa" />|<img width="48" height="48" alt="point_003496" src="https://github.com/user-attachments/assets/0bc4d7bd-e85e-43f9-a893-dbbb070f46da" />|<img width="48" height="48" alt="point_003008" src="https://github.com/user-attachments/assets/cbf23eed-5ded-4709-8e7a-0cc8eab920eb" />|

## Arch

<img width="350" alt="puc_p_48x48" src="https://github.com/user-attachments/assets/4f4bacfd-5ac1-4af6-a68b-379377f3dc49" />

## Ultra-lightweight classification model series
1. [VSDLM: Visual-only speech detection driven by lip movements](https://github.com/PINTO0309/VSDLM) - MIT License
2. [OCEC: Open closed eyes classification. Ultra-fast wink and blink estimation model](https://github.com/PINTO0309/OCEC) - MIT License
3. [PGC: Ultrafast pointing gesture classification](https://github.com/PINTO0309/PGC) - MIT License
4. [SC: Ultrafast sitting classification](https://github.com/PINTO0309/SC) - MIT License
5. [PUC: Phone Usage Classifier is a three-class image classification pipeline for understanding how people
interact with smartphones](https://github.com/PINTO0309/PUC) - MIT License
6. [HSC: Happy smile classifier](https://github.com/PINTO0309/HSC) - MIT License

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2025puc,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/PUC},
  month     = {11},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17666420},
  url       = {https://github.com/PINTO0309/puc},
  abstract  = {Phone Usage Classifier (PUC) is a three-class image classification pipeline for understanding how people
interact with smartphones.},
}
```

## Acknowledgements
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

