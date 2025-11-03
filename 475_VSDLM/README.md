# 475_VSDLM
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17494543.svg)](https://doi.org/10.5281/zenodo.17494543) ![GitHub License](https://img.shields.io/github/license/pinto0309/vsdlm) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/vsdlm)

Visual-only speech detection driven by lip movements.

There are countless situations where you can't hear the audio, and it's really frustrating.

https://github.com/user-attachments/assets/e204662f-dd54-4c19-8d9f-5a1fd8f4fab8

https://github.com/user-attachments/assets/9d68a0f0-b769-473d-8eeb-43ac7447c499

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|P|112 KB|0.9502|0.18 ms|[Download](https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_p.onnx)|
|N|176 KB|0.9586|0.31 ms|[Download](https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_n.onnx)|
|S|494 KB|0.9696|0.50 ms|[Download](https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_s.onnx)|
|C|875 KB|0.9777|0.60 ms|[Download](https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_c.onnx)|
|M|1.7 MB|0.9801|0.70 ms|[Download](https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_m.onnx)|
|L|6.4 MB|0.9891|0.91 ms|[Download](https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_l.onnx)|

## Inference

```bash
python demo_vsdlm.py \
-v 0 \
-m deimv2_dinov3_s_wholebody34_1750query_n_batch_640x640.onnx \
-vm vsdlm_l.onnx \
-ep cuda

python demo_vsdlm.py \
-v 0 \
-m deimv2_dinov3_s_wholebody34_1750query_n_batch_640x640.onnx \
-vm vsdlm_l.onnx \
-ep tensorrt
```

## Arch

<img width="300" alt="vsdlm_p" src="https://github.com/user-attachments/assets/1616215b-99f0-4c28-a1fa-b3dc647adf11" />

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2025vsdlm,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/VSDLM},
  month     = {10},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17494543},
  url       = {https://github.com/PINTO0309/vsdlm},
  abstract  = {Visual only speech detection by lip movement.},
}
```

## Acknowledgements

1. https://zenodo.org/records/3625687 - CC BY 4.0 License
2. https://spandh.dcs.shef.ac.uk/avlombard - CC BY 4.0 License
3. https://github.com/hhj1897/face_alignment - MIT License
4. https://github.com/hhj1897/face_detection - MIT License
5. https://github.com/PINTO0309/Face_Mask_Augmentation - MIT License
6. https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34 - Apache 2.0
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
7. https://github.com/PINTO0309/VSDLM - MIT License
    ```bibtex
    @software{hyodo2025vsdlm,
      author    = {Katsuya Hyodo},
      title     = {PINTO0309/VSDLM},
      month     = {10},
      year      = {2025},
      publisher = {Zenodo},
      doi       = {10.5281/zenodo.17494543},
      url       = {https://github.com/PINTO0309/vsdlm},
      abstract  = {Visual only speech detection by lip movement.},
    }
    ```
