# 501_LINEAE

LINEAE (**LINEA E**nhanced) is an experimental successor to [LINEA](https://github.com/SebastianJanampa/LINEA) aimed at improving both line detection accuracy and inference speed. It keeps the LINEA Wireframe/YorkUrban data and detector semantics, and adds selectable HGNetV2/DINOv3 backbones, progressive unfreezing, reproducible resume, XL-to-smaller line-set distillation, an optional qualified-X teacher cascade, EMA, projected feature KD/intermediate-block fusion, exact-input teacher caching, memory-efficient SDPA for every DINO variant, eval-only versioned RoPE caching, allocation-light decoder broadcasts, bounded multi-scale anchor/position caching, and deployment benchmarks.

- LINEAE-A ONNX (1.9 MB) + TensorRT FP16 + RTX3070

  https://github.com/user-attachments/assets/11627445-83cf-4f8d-a901-614378b18c96

- LINEAE-XL ONNX (388.2 MB) + TensorRT FP16 + RTX3070

  https://github.com/user-attachments/assets/008f5a87-0411-4599-b699-45d163121d9c

- LINEAE-3XL ONNX (3.8 GB) + TensorRT FP16 + RTX3070

  https://github.com/user-attachments/assets/7ef0788c-a1cf-4030-8b89-258551c625ed

## Demo

- https://github.com/PINTO0309/LINEAE
- https://github.com/PINTO0309/LINEAE/releases/tag/weights

```bash
python demo_lineae.py \
--input 0 \
--model lineae_n_1x3x640x640_1100.onnx \
--execution-provider tensorrt \
--score-threshold 0.2
```
```bash
python demo_lineae.py \
--input 0 \
--model lineae_n_1x3x640x640_1100.onnx \
--execution-provider cuda \
--score-threshold 0.2
```
```bash
python demo_lineae.py \
--input 0 \
--model lineae_n_1x3x640x640_1100.onnx \
--execution-provider cpu \
--score-threshold 0.2
```

## Parameter inventory

- WF: Wireframe, YU: YorkUrban
- [LINEAE model weights or ONNX files](https://github.com/PINTO0309/LINEAE/releases/tag/weights)

| Var | Backbone<br>(M) | Head<br>(M) | Total<br>(M) | GFLOPs | WF<br>AP<sup>5</sup> | 　<br>AP<sup>10</sup> | 　<br>AP<sup>15</sup> | YU<br>AP<sup>5</sup> | 　<br>AP<sup>10</sup> | 　<br>AP<sup>15</sup> |
| :-----: | -----------: | -----------------: | --------: | -----: | -------------: | --------------: | --------------: | -------------: | --------------: | --------------: |
| [LINEA-N](https://github.com/SebastianJanampa/LINEA) |          1.8 |                2.0 |       3.9 |   11.5 |58.7|65.0|67.9|27.3|30.5|32.5|
| [LINEA-S](https://github.com/SebastianJanampa/LINEA) |          2.2 |                6.2 |       8.4 |   29.4 |58.4|64.7|67.6|28.9|32.6|34.8|
| [LINEA-M](https://github.com/SebastianJanampa/LINEA) |          6.0 |                7.3 |      13.3 |   43.4 |59.5|66.3|69.1|30.3|34.5|36.7|
| [LINEA-L](https://github.com/SebastianJanampa/LINEA) |          13.5 |              11.5 |      25.0 |   81.5 |61.0|67.9|70.8|30.9|34.9|37.3|
| A       |          0.3 |                1.6 |       1.9 |    2.5 |48.98|57.25|60.84|27.50|34.70|38.63|
| F       |          0.7 |                1.9 |       2.6 |    4.7 |55.68|62.73|65.73|35.19|41.05|44.57|
| P       |          1.0 |                2.0 |       3.0 |   10.8 |60.51|66.36|69.21|38.24|42.85|46.03|
| N       |          1.9 |                2.1 |       3.9 |   11.7 |61.30|66.91|69.61|43.91|48.29|51.04|
| T       |          2.2 |                6.2 |       8.4 |   29.4 |63.97|69.35|71.87|52.00|55.83|58.40|
| S       |          6.0 |                5.9 |      11.9 |   39.2 |62.24|68.74|71.49|53.98|59.75|62.72|
| M       |         10.6 |                6.7 |      17.3 |   55.5 |63.86|70.05|72.51|60.68|65.87|68.32|
| L       |         23.0 |                6.7 |      29.7 |   94.5 |65.00|71.48|74.00|57.67|63.93|66.46|
| X       |         30.1 |                8.1 |      38.2 |  121.2 |65.94|72.32|74.72|62.22|68.49|71.01|
| XL      |         88.4 |                8.1 |      96.5 |  306.3 |68.71|74.24|76.36|63.42|68.17|70.38|
| 2XL     |        311.5 |               60.7 |     372.2 | 1173.6 |-|-|-|-|-|-|
| 3XL     |        853.7 |              106.8 |     960.5 | 3043.2 |72.19|76.80|78.65|70.81|74.50|76.48|

## Licensing

LINEAE is distributed under the root [Apache License 2.0](LICENSE).

## Cited / Acknowledgement
- https://github.com/SebastianJanampa/LINEA - Apache License 2.0
  ```bibtex
  @misc{janampa2025linea,
    title={LINEA: Fast and Accurate Line Detection Using Scalable Transformers},
    author={Sebastian Janampa and Marios Pattichis},
    year={2025},
    eprint={2505.16264},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2505.16264},
  }
  ```
- https://github.com/Intellindust-AI-Lab/DEIMv2 - Apache License 2.0
  ```bibtex
  @article{huang2025deimv2,
    title={Real-Time Object Detection Meets DINOv3},
    author={Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
    journal={arXiv},
    year={2025}
  }
  ```
- https://github.com/Peterande/D-FINE - Apache License 2.0
  ```bibtex
  @misc{peng2024dfine,
    title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
    author={Yansong Peng and Hebei Li and Peixi Wu and Yueyi Zhang and Xiaoyan Sun and Feng Wu},
    year={2024},
    eprint={2410.13842},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
  }
  ```
- https://github.com/facebookresearch/dinov3 - DINOv3 License
  ```bibtex
  @misc{simeoni2025dinov3,
    title={{DINOv3}},
    author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
    year={2025},
    eprint={2508.10104},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2508.10104},
  }
  ```
- https://github.com/huangkuns/wireframe - MIT License
  ```bibtex
  @InProceedings{wireframe_cvpr18,
    author = {Kun Huang and Yifan Wang and Zihan Zhou and Tianjiao Ding and Shenghua Gao and Yi Ma},
    title = {Learning to Parse Wireframes in Images of Man-Made Environments},
    booktitle = {CVPR},
    month = {June},
    year = {2018}
  }
  ```
- https://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/
  ```bibtex
  @Inbook{Denis2008,
    author="Denis, Patrick and Elder, James H. and Estrada, Francisco J.",
    title="Efficient Edge-Based Methods for Estimating Manhattan Frames in Urban Imagery",
    bookTitle="Computer Vision -- ECCV 2008: 10th European Conference on Computer Vision, Marseille, France, October 12-18, 2008, Proceedings, Part II",
    year="2008",
    publisher="Springer Berlin Heidelberg",
    pages="197--210",
    isbn="978-3-540-88688-4",
    doi="10.1007/978-3-540-88688-4_15"
  }
  ```
- https://github.com/PINTO0309/gazelle-dinov3 - MIT License
  ```bibtex
  @software{Hyodo_2025_gazelle_dinov3,
    author    = {Katsuya Hyodo},
    title     = {gazelle-dinov3: Gaze-LLE-DINOv3},
    year      = {2025},
    month     = {oct},
    publisher = {Zenodo},
    version   = {1.0.0},
    doi       = {10.5281/zenodo.17413165},
    url       = {https://github.com/PINTO0309/gazelle-dinov3},
    abstract  = {A model for activating human gaze regions using heat maps, built with DINOv3.},
  }
  ```
