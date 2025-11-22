# 479_PUC
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17666420.svg)](https://doi.org/10.5281/zenodo.17666420) ![GitHub License](https://img.shields.io/github/license/pinto0309/PUC)
 [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/puc)

Phone Usage Classifier (PUC) is a three-class image classification pipeline for understanding how people
interact with smartphones. **Perhaps the model is looking at our `hands`, not our `smartphones`. This model is a complete failure, but it shows how humans fail to look at the small details when making judgments.**

- `classid=0` (`no_action`): No interaction with a smartphone.
- `classid=1` (`point`): Pointing the smartphone towards the camera.
- `classid=2` (`point_somewhere`): Pointing the smartphone somewhere other than the camera.

https://github.com/user-attachments/assets/18acf290-63b6-40ba-a38c-a5712dedc19c

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|P|115 KB|0.9628|0.13 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_p_32x24.onnx)|
|N|176 KB|0.9754|0.24 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_n_32x24.onnx)|
|T|280 KB|0.9923|0.31 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_t_32x24.onnx)|
|S|495 KB|0.9975|0.35 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_s_32x24.onnx)|
|C|876 KB|0.9979|0.47 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_c_32x24.onnx)|
|M|1.7 MB|0.9985|0.55 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_m_32x24.onnx)|
|L|6.4 MB|0.9986|0.73 ms|[Download](https://github.com/PINTO0309/PUC/releases/download/onnx/puc_l_32x24.onnx)|

## Data sample

|1|2|3|4|
|:-:|:-:|:-:|:-:|
|<img width="24" height="32" alt="000000084193_022002_0" src="https://github.com/user-attachments/assets/9e661b3d-f5ee-4a4a-bcee-8d28d6ac020a" />|<img width="24" height="32" alt="no_action1_004005_0" src="https://github.com/user-attachments/assets/5e26aa1d-f849-47d2-ae73-88ec2c4bedd9" />|<img width="24" height="32" alt="point1_001301_1" src="https://github.com/user-attachments/assets/fe84a427-8d86-45f1-b77e-e4d6778b1a23" />|<img width="24" height="32" alt="point_somewhere4_000156_2" src="https://github.com/user-attachments/assets/3682f7e5-26a0-4e70-a38b-93c60c3f5a31" />|

## Arch

<img width="350" alt="puc_p_32x24" src="https://github.com/user-attachments/assets/82fd2a31-d6b5-4bbb-a099-ac51237d145c" />

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

