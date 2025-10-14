# 473_HISDF

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

HISDF (Human Instance, Skeleton, and Depth Fusion) is a unified model that fuses human instance segmentation, skeletal structure estimation, and depth prediction to achieve holistic human perception from visual input.

Merging DEIMv2 and DepthAnythingV2, RHIS.

## Demo - X size Monolithic Model
- RTX3070 8GB
- Object Detection (34 classes)
- Joint Detection
- Attribute estimation (Generation, Gender, Wheelchair, Crutches)
- Head Orientation Estimation (8 directions)
- Absolute distance detection
- Depth estimation
- Instance Segmentation
- Binary Segmentation

  https://github.com/user-attachments/assets/2fcad55b-775c-46b7-bcff-1132baa7a89d

  https://github.com/user-attachments/assets/17544727-f971-4d76-91d6-096bc428506f

  https://github.com/user-attachments/assets/6c86e438-2953-4e69-8b29-e1f77846ad03

  https://github.com/user-attachments/assets/15f35c51-9aef-43dc-a128-b34d3610825e

## Demo - S size Monolithic Model
- RTX3070 8GB
- Object Detection (34 classes)
- Joint Detection
- Attribute estimation (Generation, Gender, Wheelchair, Crutches)
- Head Orientation Estimation (8 directions)
- Absolute distance detection
- Depth estimation
- Instance Segmentation
- Binary Segmentation

  https://github.com/user-attachments/assets/e636893c-b51a-4616-9bb5-d0a51a02baf6

## Features

- Multitask inference that combines person detection, attribute estimation, skeleton keypoints, and per-pixel depth from a single forward pass.
- Instance segmentation masks and keypoint overlays with stable colouring driven by a lightweight SORT-style tracker.
- Optional depth-map and mask compositing overlays to visualise the fused predictions in real time or on saved frames.
- Support for CPU, CUDA, and TensorRT execution via ONNX Runtime providers.
- Utilities for exporting detections to YOLO format and automatically persisting rendered frames or videos.

## Repository Layout

```
├── demo_hisdf_onnx_34.py        # Main demo / visualisation entry point
├── merge_preprocess_onnx_depth_seg.py
├── *.onnx                       # Trained HISDF, depth, and post-processing models
├── pyproject.toml               # Python package metadata and dependencies
├── uv.lock                      # Reproducible environment lock file
└── README.md
```

## Model Zoo

The repository ships with several ONNX artefacts:　https://github.com/PINTO0309/HISDF/releases

- `deimv2_dinov3_x_wholebody34_*.onnx`: core HISDF detectors with varying query counts.
- `depth_anything_v2_small_*.onnx`: depth backbones at different input resolutions.
- `postprocess_*` / `preprocess_*`: helper networks for resizing, segmentation, or depth refinement.
- `bboxes_processor.onnx`: post-processing utilities for bounding-box outputs.

Pick the variant that best matches your latency/accuracy trade-offs. The demo defaults to `deimv2_depthanythingv2_instanceseg_1x3xHxW.onnx`.

## Installation

If you use `uv`, simply run `uv sync` to materialise the locked environment.

```bash
git clone https://github.com/PINTO0309/HISDF && cd HISDF
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
export PYTHONWARNINGS="ignore"
```


### GPU / TensorRT notes

- CUDA provider needs the matching CUDA toolkit and cuDNN runtime.
- TensorRT execution (`--execution_provider tensorrt`) requires a built engine cache directory with write access.

## Quick Start

### Run on a webcam

```bash
python demo_hisdf_onnx_34.py \
--model deimv2_depthanythingv2_instanceseg_1x3xHxW.onnx \
--video 0 \
--execution_provider cuda
```

### Run on an image directory

```bash
python demo_hisdf_onnx_34.py \
--images_dir ./samples \
--disable_waitKey
```

Rendered images will be written to `./output` together with YOLO annotations when `--output_yolo_format_text` is enabled.

## Key Command-Line Flags

| Flag | Description |
| --- | --- |
| `--video`, `--images_dir` | Choose between streaming input and batch image inference (one is required). |
| `--execution_provider {cpu,cuda,tensorrt}` | Select the ONNX Runtime backend. |
| `--object_socre_threshold` | Detection confidence for object classes. |
| `--attribute_socre_threshold` | Confidence threshold for attribute heads. |
| `--keypoint_threshold` | Minimum score for bone/keypoint visualisation. |
| `--disable_*` toggles | Turn off attribute-specific rendering (generation, gender, handedness, head pose). |
| `--disable_video_writer` | Skip MP4 recording when reading from a video source. |
| `--enable_face_mosaic` | Pixelate face detections to preserve privacy. |
| `--output_yolo_format_text` | Export YOLO labels alongside rendered frames. |

Run `python demo_hisdf_onnx_34.py --help` for the full list.

## Visualisation Controls

While the demo window is active you can toggle features with the keyboard:

- `n` Generation (adult/child) display
- `g` Gender colouring
- `p` Head pose labelling
- `h` Handedness identification
- `k` Keypoint drawing mode (`dot` → `box` → `both`)
- `f` Face mosaic
- `b` Skeleton visibility
- `d` Depth-map overlay
- `i` Instance mask overlay
- `m` Head-distance measurement

Persistent track IDs for person boxes are drawn outside the bounding boxes. Colours are locked per track and shared with the instance masks to avoid flicker when detection ordering changes.

## Development Tips

- `demo_hisdf_onnx_34.py` is structured around a `HISDF` model wrapper that encapsulates preprocessing, inference, and postprocessing for all tasks.
- The lightweight `SimpleSortTracker` keeps body detections stable across frames; tune the IoU threshold or `max_age` if you encounter ID churn.
- Depth overlays rely on OpenCV colormaps and the segmentation mask to blend only foreground pixels.
- Use `python -m compileall demo_hisdf_onnx_34.py` to perform a quick syntax check after edits.
- Model Inputs and Outputs
  - Inputs
    |Name|Type|Note|
    |:-|:-|:-|
    |`input_bgr`|`float32[1, 3, H, W]`|BGR image|
  - Outputs
    |Name|Type|Note|
    |:-|:-|:-|
    |`bbox_classid_xyxy_score`|`float32[num_rois, 6]`|Bounding box of object detection result. `[boxes, [classid, x1, y1, x2, y2, score]]`. Coordinates normalized to 0.0-1.0|
    |`depth`|`float32[1, 1, H, W]`|A depth map of the same size as the input image.|
    |`binary_masks`|`float32[1, 1, H, W]`|A binary mask of the same size as the input image.|
    |`instance_masks`|`float32[num_rois, 1, 160, 120]`|The number of instance segmentation masks is the same as the number of bodies (ROIs) in the object detection result. 160x120 size based on RHIS model input resolution 640x640. The ROI needs to be rescaled depending on the resolution of the input image.|
  - Sample

    <img width="827" height="525" alt="image" src="https://github.com/user-attachments/assets/855e9269-ee82-4482-a1a0-0295772d025c" />

  - Class IDs
    ```
    ┏━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━┓
    ┃ ID┃Name                 ┃ ID┃Name                 ┃
    ┡━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━┩
    │  0│body                 │ 20│ear                  │
    │  1│adult                │ 21│collarbone           │
    │  2│child                │ 22│shoulder             │
    │  3│male                 │ 23│solar_plexus         │
    │  4│female               │ 24│elbow                │
    │  5│body_with_wheelchair │ 25│wrist                │
    │  6│body_with_crutches   │ 26│hand                 │
    │  7│head                 │ 27│hand_left            │
    │  8│front                │ 28│hand_right           │
    │  9│right-front          │ 29│abdomen              │
    │ 10│right-side           │ 30│hip_joint            │
    │ 11│right-back           │ 31│knee                 │
    │ 12│back                 │ 32│ankle                │
    │ 13│left-back            │ 33│foot                 │
    │ 14│left-side            │   │                     │
    │ 15│left-front           │   │                     │
    │ 16│face                 │   │                     │
    │ 17│eye                  │   │                     │
    │ 18│nose                 │   │                     │
    │ 19│mouth                │   │                     │
    └───┴─────────────────────┴───┴─────────────────────┘
    ```

## Troubleshooting

- **No window appears**: Ensure an X server is available, or run headless by setting `--disable_waitKey` and writing frames to disk.
- **ONNX Runtime errors**: Confirm the selected execution provider matches your hardware and that all provider-specific dependencies are installed.
- **Incorrect colours or masks**: Make sure the instance segmentation overlay is enabled (`i` key) and verify that your model outputs masks.
- **Slow FPS**: Disable depth and mask overlays, lower the input resolution, or switch to TensorRT.

## LICENSE
This project is licensed under the Apache License Version 2.0 License.

Refer to [LICENSE](./LICENSE) for the full terms governing the use of the code and bundled models.

## Citation
If you find this project useful, please consider citing:

```bibtex
@software{hisdf,
  title={HISDF (Human Instance, Skeleton, and Depth Fusion)},
  author={Katsuya Hyodo},
  version={1.0.0},
  year={2025},
  doi={10.5281/zenodo.17274823},
  url={https://github.com/PINTO0309/HISDF},
  note={HISDF: unified human instance, skeleton, and depth fusion model.}
}
```

## Acknowledgments

- https://github.com/Intellindust-AI-Lab/DEIMv2
  ```bibtex
  @article{huang2025deimv2,
    title={Real-Time Object Detection Meets DINOv3},
    author={Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
    journal={arXiv},
    year={2025}
  }
  ```
- https://github.com/DepthAnything/Depth-Anything-V2
  ```bibtex
  @article{depth_anything_v2,
    title={Depth Anything V2},
    author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
    journal={arXiv:2406.09414},
    year={2024}
  }

  @inproceedings{depth_anything_v1,
    title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data},
    author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
    booktitle={CVPR},
    year={2024}
  }
  ```
- https://github.com/PINTO0309/human-instance-segmentation
- https://github.com/PINTO0309/DEIMv2
