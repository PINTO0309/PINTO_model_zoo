# 500_ScaleLSD

<div align="center">

# ScaleLSD: Scalable Deep Line Segment Detection Streamlined

<a href="https://ant-research.github.io/scalelsd"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a>&ensp;<a href="https://arxiv.org/abs/2506.09369"><img src="https://img.shields.io/badge/ArXiv-2506.09369-brightgreen"></a>&ensp;<a href="https://huggingface.co/cherubicxn/scalelsd"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a>



[Zeran Ke](https://calmke.github.io/)<sup>1,2</sup>, [Bin Tan](https://icetttb.github.io/)<sup>2</sup>, [Xianwei Zheng](https://jszy.whu.edu.cn/zhengxianwei/zh_CN/index.htm)<sup>1</sup>,  [Yujun Shen](https://shenyujun.github.io/)<sup>2</sup>, [Tianfu Wu](https://research.ece.ncsu.edu/ivmcl/)<sup>3</sup>, [Nan Xue](https://xuenan.net/)<sup>2†</sup>

<sup>1</sup>Wuhan University &ensp;&ensp;<sup>2</sup>Ant Group&ensp;&ensp;<sup>3</sup>NC State University

</div>

![teaser](assets/teaser.jpg)

## Original
https://github.com/ant-research/scalelsd

## ONNX custom
https://github.com/PINTO0309/scalelsd

## ONNX export

`export_onnx.py` exports a tensor-only, fixed-capacity approximation of the
ScaleLSD inference post-processing. Its input is RGB Float32 in `[0, 1]` with
shape `[B, 3, H, W]`; grayscale conversion is part of the model. `H` and `W`
are fixed at export time and must be multiples of 32.

Place the ScaleLSD checkpoint at the default model path shown below, then run:

```bash
uv run python export_onnx.py
```

The default command writes `models/scalelsd-vitbase-v1-train-sa1b.onnx`, uses
opset 17 and batch size 1, validates it with `onnx.checker`, and simplifies it
with `onnxsim`. Examples of the main variants are:

```bash
# Select another opset and fixed batch size.
uv run python export_onnx.py --opset 18 --batch-size 2 -o models/scalelsd_b2.onnx

# Make only axis 0 dynamic. Spatial and detection-capacity axes remain fixed.
uv run python export_onnx.py --dynamic-batch -o models/scalelsd_dynamic.onnx

# Retain the checked model without running onnxsim.
uv run python export_onnx.py --disable-onnxsim

# Export two RGB images through ScaleLSD, SuperPoint, and GlueStick as one model.
uv run python export_onnx.py --end2end -o models/scalelsd_end2end.onnx
```

`--width` and `--height` default to 512. Detection capacities are controlled
by `--num-junctions` (512), `--max-lines` (300), and `--max-keypoints` (1000,
end-to-end only). `--dynamic-batch` assigns the shared symbolic name `batch`
only to axis 0 of every input and output; without it, `--batch-size` is fixed.

The detector model exposes:

| Name | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `image` | Float32 | `[B,3,H,W]` | RGB image in `[0,1]` |
| `junctions` | Float32 | `[B,J,2]` | input-image `x,y` coordinates |
| `junction_scores` | Float32 | `[B,J]` | junction confidence |
| `junction_valid` | Bool | `[B,J]` | valid padded entry mask |
| `lines` | Float32 | `[B,L,4]` | input-image `x1,y1,x2,y2` coordinates |
| `line_scores` | Float32 | `[B,L]` | supporting-field count |
| `line_valid` | Bool | `[B,L]` | valid padded entry mask |

With `--end2end`, the inputs are `image0` and `image1`. The outputs are
`lines0`, `line_scores0`, `line_valid0`, the corresponding three `*1` tensors,
then `line_matches0`, `line_match_scores0`, `line_matches1`, and
`line_match_scores1`, all with fixed line capacity `L`. Match tensors are
Int64 indices into the opposite image's line array. An invalid match has index
`-1` and score `0`.

The end-to-end graph uses deterministic fixed top-k junctions/keypoints instead
of the variable-length OpenCV, DBSCAN, and random-padding path. GlueStick is
pinned to commit `7d816730ef939caa1c61e2564eceda77304874fa`. The official GlueStick
and SuperPoint weights are downloaded to the Torch Hub cache when missing. Pass
`--gluestick-weights` or `--superpoint-weights` to load a local file first; if
the specified file does not exist, the official weights are downloaded and
saved there. The ScaleLSD checkpoint is always required locally.

Export is performed to a temporary file, checked, optionally simplified, and
checked again before replacing the requested output. If simplification fails,
the checked raw graph is kept next to the destination as
`*.unsimplified.onnx`, and the command exits non-zero.

### ONNX-only demo

`demo_scalelsd.py` runs the exported detector directly with ONNX Runtime,
NumPy, and OpenCV. It does not import the training/inference package. With no
input arguments it uses the v2 ONNX model and `assets/indoor.jpg`:

```bash
uv run python demo_scalelsd.py
```

Use `--disable-display` in a headless environment. The demo also detects an
OpenCV build without a GUI backend and automatically falls back to saving only.
Rendered images and WireframeGraph-compatible JSON files are written to
`output/scalelsd_onnx`.

```bash
# A single image on CPU.
uv run python demo_scalelsd.py \
--image assets/indoor.jpg \
--execution-provider cpu \
--disable-display

# Every supported image directly inside a directory.
uv run python demo_scalelsd.py --images-dir assets/figs

# A video file, or camera index 0. Press q or Esc to stop.
uv run python demo_scalelsd.py --video input.mp4
uv run python demo_scalelsd.py --video 0

# Select the v1 model or TensorRT.
uv run python demo_scalelsd.py \
--model onnx/scalelsd-vitbase-v2-train-sa1b.onnx \
--execution-provider tensorrt

# Use TensorRT FP32 for the whole graph when maximum numerical agreement is
# more important than speed.
uv run python demo_scalelsd.py \
--model onnx/scalelsd-vitbase-v2-train-sa1b.onnx \
--execution-provider tensorrt \
--tensorrt-precision fp32
```

The provider order is TensorRT → CUDA → CPU, CUDA → CPU, or CPU-only according
to `--execution-provider`. TensorRT uses an FP16 backbone and CUDA FP32 for the
precision-sensitive fixed-size post-processing by default. TensorRT engine
caches are written beside the selected ONNX file. Cache prefixes include the
ONNX filename and precision mode, preventing v1/v2 and mixed/FP32 engines from
colliding. Annotated videos are saved as MP4; static images also produce JSON
in the original image coordinate system.

#### Why the TensorRT post-processing stays in FP32

Running the entire exported detector in TensorRT FP16 is numerically unsafe.
The fixed-size line post-processing computes squared distances from field
coordinates that can reach approximately 255. Consequently, the intermediate
term `x² + y²` can reach approximately 130,050, exceeding the largest finite
FP16 value of 65,504. Once these distance calculations overflow to infinity,
many line endpoints can collapse onto the same junction during nearest-junction
assignment. In our reproduction this made a line-support score increase from
249 with CUDA FP32 to an invalid 2,048 with full TensorRT FP16, visibly
corrupting the output.

The default `mixed` mode therefore keeps the backbone in TensorRT FP16 while
executing distance calculation and graph aggregation in CUDA FP32. Use
`--tensorrt-precision fp32` when the entire TensorRT graph should run in FP32;
full-graph TensorRT FP16 is intentionally not exposed by this demo.


## 📝 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{ScaleLSD,
    title = {ScaleLSD: Scalable Deep Line Segment Detection Streamlined},
    author = {Zeran Ke and Bin Tan and Xianwei Zheng and Yujun Shen and Tianfu Wu and Nan Xue},
    booktitle = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
    year = {2025},
}
```
