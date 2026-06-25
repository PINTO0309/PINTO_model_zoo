# 492_Efficient-FIQA

- **🏆 🥇 Winner Solution for [ICCV VQualA 2025 Face Image Quality Assessment Challenge](https://codalab.lisn.upsaclay.fr/competitions/23017)**

## 🎯 Introduction

**Face Image Quality Assessment (FIQA)** is crucial for various face-related applications such as face recognition, face detection, and biometric systems. While significant progress has been made in FIQA research, the computational complexity remains a key bottleneck for real-world deployment.

This repository presents **Efficient-FIQA**, a novel approach that achieves state-of-the-art performance with extremely low computational overhead through:

- **🔬 Self-training Strategy**: Enhances teacher model capacity using pseudo-labeled data
- **🎓 Knowledge Distillation**: Transfers knowledge from powerful teacher to lightweight student
- **⚡ Efficient Architecture**: Student model achieves comparable performance with minimal computational cost

### 🏆 Key Achievements

- **🥇 1st Place** in ICCV VQualA 2025 FIQA Challenge

---

## 🏆 Challenge Results

| Rank | Team | Score | GFLOPs | Params (M) |
|:----:|------|:-----:|:------:|:----------:|
| 🥇 **1** | **ECNU-SJTU VQA Team (Ours)** | **0.9664** | **0.3313** | **1.1796** |
| 2 | MediaForensics | 0.9624 | 0.4687 | 1.5189 |
| 3 | Next | 0.9583 | 0.4533 | 1.2224 |
| 4 | ATHENAFace | 0.9566 | 0.4985 | 2.0916 |
| 5 | NJUPT-IQA-Group | 0.9547 | 0.4860 | 3.7171 |
| 6 | ECNU VIS Lab | 0.9406 | 0.4923 | 3.2805 |

*Score = (SRCC + PLCC) / 2*

## ONNX Runtime Inference

`demo.py` runs FIQA_EdgeNeXt_XXS ONNX models without importing PyTorch or TorchVision.

```bash
# Fixed 1x3x352x352 ONNX, CPU backend
uv run python demo.py \
--backend cpu \
--onnx_file FIQA_EdgeNeXt_XXS_1x3x352x352.onnx \
--image_file demo_images/z06399_368x488_0.3676.png

# Dynamic H/W ONNX, CPU backend
uv run python demo.py \
--backend cpu \
--onnx_file FIQA_EdgeNeXt_XXS_1x3xHxW.onnx \
--image_file demo_images/z06399_368x488_0.3676.png \
--height 320 \
--width 384

# CUDA backend (requires an ONNX Runtime build with CUDAExecutionProvider)
uv run python demo.py \
--backend cuda \
--gpu_id 0 \
--onnx_file FIQA_EdgeNeXt_XXS_1x3x352x352.onnx \
--image_file demo_images/z06399_368x488_0.3676.png

# TensorRT backend (requires TensorrtExecutionProvider)
uv run python demo.py \
--backend tensorrt \
--gpu_id 0 \
--onnx_file FIQA_EdgeNeXt_XXS_1x3x352x352.onnx \
--image_file demo_images/z06399_368x488_0.3676.png
```

The default backend is `cuda`. If the requested ONNX Runtime execution provider is not available, `demo.py` exits with an explicit error instead of falling back to CPU.

## Sample

```bash
uv run python demo.py \
--backend cuda \
--onnx_file onnx/FIQA_EdgeNeXt_XXS_1x3x352x352.onnx \
--image_file demo_images/z06399_368x488_0.3676.png

The quality score of the image demo_images/z06399_368x488_0.3676.png is 0.3676
```

## Cited

https://github.com/sunwei925/Efficient-FIQA