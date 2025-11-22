# 480_HSC
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17670546.svg)](https://doi.org/10.5281/zenodo.17670546) ![GitHub License](https://img.shields.io/github/license/pinto0309/HSC) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/hsc)

Happy smile classifier. The estimation is done on the entire head, 48x48 pixels, rather than the face.

https://github.com/user-attachments/assets/f4a68c3a-ed66-4823-a910-e5719a665821

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|P|115 KB|0.4841|0.23 ms|[Download](https://github.com/PINTO0309/HSC/releases/download/onnx/hsc_p_48x48.onnx)|
|N|176 KB|0.5849|0.41 ms|[Download](https://github.com/PINTO0309/HSC/releases/download/onnx/hsc_n_48x48.onnx)|
|T|280 KB|0.6701|0.52 ms|[Download](https://github.com/PINTO0309/HSC/releases/download/onnx/hsc_t_48x48.onnx)|
|S|495 KB|0.7394|0.64 ms|[Download](https://github.com/PINTO0309/HSC/releases/download/onnx/hsc_s_48x48.onnx)|
|C|876 KB|0.7344|0.69 ms|[Download](https://github.com/PINTO0309/HSC/releases/download/onnx/hsc_c_48x48.onnx)|
|M|1.7 MB|0.8144|0.85 ms|[Download](https://github.com/PINTO0309/HSC/releases/download/onnx/hsc_m_48x48.onnx)|
|L|6.4 MB|0.8293|1.03 ms|[Download](https://github.com/PINTO0309/HSC/releases/download/onnx/hsc_l_48x48.onnx)|

## Data sample

|1|2|3|4|
|:-:|:-:|:-:|:-:|
|<img width="48" height="48" alt="image" src="https://github.com/user-attachments/assets/c14e1566-6a2c-49fd-8835-2cdbafd9959c" />|<img width="48" height="48" alt="image" src="https://github.com/user-attachments/assets/a6dc9668-fa5b-46a3-8787-361dd7371e79" />|<img width="48" height="48" alt="image" src="https://github.com/user-attachments/assets/9e29ae46-c7e6-437f-8b5c-6c235478b2e5" />|<img width="48" height="48" alt="image" src="https://github.com/user-attachments/assets/165ff7e1-caf1-4948-93cb-2d45c93e4c66" />|

## Arch

<img width="350" alt="hsc_p_48x48" src="https://github.com/user-attachments/assets/b3c79843-004d-4b12-a51a-34d707242f6c" />

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
@software{hyodo2025hsc,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/HSC},
  month     = {11},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17670546},
  url       = {https://github.com/PINTO0309/hsc},
  abstract  = {Happy smile classifier.},
}
```

## Acknowledgments

- https://github.com/microsoft/FERPlus: MIT License
  ```bibtex
  @inproceedings{BarsoumICMI2016,
      title={Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution},
      author={Barsoum, Emad and Zhang, Cha and Canton Ferrer, Cristian and Zhang, Zhengyou},
      booktitle={ACM International Conference on Multimodal Interaction (ICMI)},
      year={2016}
  }
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
