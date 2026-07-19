# 498_DocRes

<p align="center">
<img src="images/motivation.jpg" width="400">
</p>


- original

  https://github.com/ZZZHANG-jx/DocRes

- ONNX custom

  https://github.com/PINTO0309/DocRes

## Environment setup

This project uses [uv](https://docs.astral.sh/uv/) and a project-local virtual
environment. Python 3.12.12 and every Python dependency are pinned; PyTorch
2.11.0 and torchvision 0.26.0 use the CUDA 12.8 wheel index. The lock targets
Linux x86_64 with an NVIDIA GPU.

```bash
uv sync --frozen
source .venv/bin/activate
```

`uv sync` installs the pinned Python version when it is not already available
and creates `.venv` automatically. Activating the environment is optional when
using `uv run`.

To intentionally update the lock after changing a version in `pyproject.toml`:

```bash
uv lock --upgrade
uv sync
```

## ONNX export

Export both local checkpoints in one command:

```bash
uv run --frozen python export_onnx.py
```

This creates `onnx/docres.onnx` and `onnx/mbd.onnx`. By default,
both models use ONNX opset 17 and a fixed batch size of 1. The DocRes input
shape is `[1, 6, 256, 256]`, and the MBD input shape is
`[1, 3, 448, 448]`. Only the raw neural networks are exported; image
preprocessing, DTSPrompt construction, and task-specific postprocessing remain
in the application code.

After each export, `onnxsim` runs automatically and replaces the raw ONNX file
with its simplified, equivalence-checked form. Disable this step only when
needed:

```bash
uv run --frozen python export_onnx.py --disable-onnxsim
```

Dynamic-batch DocRes exports use a conservative `onnxsim` mode that preserves
the symbolic batch axis while still applying safe constant folding.

Use a different opset with `--opset`:

```bash
uv run --frozen python export_onnx.py --opset 18
```

The batch dimension remains fixed to 1 unless `--dynamic-batch` is explicitly
specified:

```bash
uv run --frozen python export_onnx.py --dynamic-batch
```

In dynamic mode, the input and output batch axes of both models are named
`batch_size`; channel and spatial dimensions remain fixed. Checkpoint paths and
the output directory can be changed with `--docres-checkpoint`,
`--mbd-checkpoint`, and `--output-dir`. Run `python export_onnx.py --help` for
the complete CLI reference.

### Chained ONNX validation

The two exported models can be validated without composing a separate
end-to-end ONNX graph. `validate_onnx_chain.py` opens `mbd.onnx` and
`docres.onnx` as independent ONNX Runtime sessions and executes the learned
part of the dewarping pipeline in sequence:

```text
Input BGR image
  -> MBD preprocessing and mbd.onnx
  -> mask morphology and dewarping DTSPrompt construction
  -> docres.onnx
  -> flow smoothing and OpenCV remap
  -> dewarped BGR image
```

Run the chain and optionally compare it with an existing PyTorch result:

```bash
uv run --frozen python validate_onnx_chain.py \
  --input input/for_dewarping.png \
  --reference restorted/for_dewarping_dewarping.png \
  --json-output onnx_results/for_dewarping_metrics.json
```

The restored image is written to
`onnx_results/<input-name>_dewarped.png` by default. Use `--output` to change
the path, `--mask-output` to inspect the thresholded MBD mask, and
`--prompt-output` to save the three-channel DTSPrompt. `--provider cpu` selects
the CPU Execution Provider instead of CUDA. The script validates both model
interfaces before inference and rejects unexpected tensor types, channels, or
spatial dimensions.

MBD and DocRes are chained only for dewarping because MBD supplies the learned
document mask used by that task. Deshadowing, appearance enhancement,
deblurring, and binarization use DocRes with deterministic prompts and do not
consume MBD output. The current fixed 256 x 256 `docres.onnx` is therefore
sufficient for this two-model chain validation. The model tensor resolutions
already match the PyTorch dewarping implementation exactly: MBD receives
`[1, 3, 448, 448]`, and DocRes receives `[1, 6, 256, 256]`. Increasing either
resolution would move the validation away from the PyTorch reference rather
than make it closer.

The chained CUDA run was tested with the 4032 x 3024
`input/for_dewarping.png` example against the saved PyTorch result:

| Metric | Result |
| --- | ---: |
| PSNR | 40.771 dB |
| SSIM | 0.98838 |
| Mean absolute error | 0.999 pixel levels |
| MBD inference | 0.307 s |
| DocRes inference | 8.152 s |
| Complete pipeline after session setup | 8.722 s |

A second test passed exactly the same preprocessed tensors to PyTorch and ONNX
in one process to isolate backend differences. The MBD output mean absolute
error was `1.26e-4`; after morphology and thresholding, only `0.0070%` of mask
pixels differed. The DocRes output mean absolute error was `1.15e-5`. Comparing
the complete ONNX chain with the simultaneously executed PyTorch chain gave
53.137 dB PSNR, 0.99876 SSIM, and a 0.176-level mean absolute image error.

The small difference from the saved reference is expected from backend-level
floating-point and interpolation differences. No combined `end2end.onnx` is
required or generated by this validation workflow, and the two model files can
continue to be updated and tested independently.

## Inference
1. Put MBD model weights [mbd.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK) to `./data/MBD/checkpoint/`
2. Put DocRes model weights [docres.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK) to `./checkpoints/`
3. Run the following script and the results will be saved in `./restorted/`. We have provided some distorted examples in `./input/`.
```bash
uv run --frozen python inference.py --im_path ./input/for_dewarping.png --task dewarping --save_dtsprompt 1
```

- `--im_path`: the path of input document image
- `--task`: task that need to be executed, it must be one of _dewarping_, _deshadowing_, _appearance_, _deblurring_, _binarization_, or _end2end_
- `--save_dtsprompt`: whether to save the DTSPrompt

## Evaluation

1. Dataset preparation, see [dataset instruction](./data/README.md)
2. Put MBD model weights [mbd.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK) to `data/MBD/checkpoint/`
3. Put DocRes model weights [docres.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK) to `./checkpoints/`
2. Run the following script
```bash
uv run --frozen python eval.py --dataset realdae
```
- `--dataset`: dataset that need to be evaluated, it can be set as _dir300_, _kligler_, _jung_, _osr_, _docunet\_docaligner_, _realdae_, _tdd_, and _dibco18_.

## Training
1. Dataset preparation, see [dataset instruction](./data/README.md)
2. Specify the datasets_setting within `train.py` based on your dataset path and experimental setting.
3. Run the following script
```bash
bash start_train.sh
```

## Citation
```
@inproceedings{zhangdocres2024,
Author = {Jiaxin Zhang, Dezhi Peng, Chongyu Liu , Peirong Zhang and Lianwen Jin},
Booktitle = {In Proceedings of the IEEE/CV Conference on Computer Vision and Pattern Recognition},
Title = {{DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks}},
Year = {2024}}
```

## Model inventory

The standard inference pipeline uses two learned models. No language model,
diffusion model, Hugging Face model, or external model service is used. The
ONNX files in this project are exported representations of the same two local
checkpoints, not additional learned models.

```text
Input RGB image
  -> task-specific DTSPrompt generation
  -> concatenate RGB (3 channels) and DTSPrompt (3 channels)
  -> DocRes / Restormer (6 input channels)
  -> restored image, displacement map, or binarization result
```

### DocRes restoration model

`checkpoints/docres.pkl` contains the main multi-task DocRes model. Its
architecture is a modified [Restormer](https://arxiv.org/abs/2111.09881) with
six input channels, three output channels, and 15,203,680 trainable parameters.
The same model handles dewarping, deshadowing, appearance enhancement,
deblurring, and binarization. The task is selected through the three-channel
[DTSPrompt described in the DocRes paper](https://arxiv.org/abs/2405.04408),
not through separate task-specific model weights.

The inspected checkpoint has the following properties:

- File size: 183,260,711 bytes
- Saved epoch: 28
- Contents: model state and optimizer state
- SHA-256: `1d6a89d754fe1e58ffd1865eab0ef3f03344798d39197b2d9a77ce4fbc8c02fd`

### MBD document-mask model

`data/MBD/checkpoint/mbd.pkl` is used only to produce the document mask for the
dewarping DTSPrompt. It is a
[DeepLabv3+-style](https://arxiv.org/abs/1802.02611) binary segmentation model
with a ResNet-101 backbone, ASPP, a decoder, output stride 16, and 59,339,169
trainable parameters. The input is resized to 448 x 448, and the predicted mask
is resized back to the original image resolution.

The inspected checkpoint has the following properties:

- File size: 712,981,585 bytes
- Saved epoch: 116
- Contents: model state and optimizer state
- SHA-256: `7c2dc15a6b0e613adf7c3a794891f44caef544b92c5898ac610ae689e9cd9085`

The checkpoint files are larger than their inference-only model states because
they also contain optimizer state.

### Models used by task

| Task | DocRes | MBD | DTSPrompt source |
| --- | --- | --- | --- |
| Dewarping | Yes | Yes | Coordinate map and learned document mask |
| Deshadowing | Yes | No | Morphological background estimate |
| Appearance enhancement | Yes | No | Illumination-normalized image |
| Deblurring | Yes | No | Sobel gradient map |
| Binarization | Yes | No | Sauvola threshold, gradient, and binary map |
| End-to-end | Three passes | One pass | Dewarping, then deshadowing, then appearance enhancement |

Except for the MBD-generated document mask, DTSPrompts are produced with
deterministic OpenCV and NumPy operations rather than additional learned
models.

### Included but inactive model implementations

The `data/MBD/model/` directory also contains alternative DeepLab backbones
(Xception, DRN-D-54, and MobileNetV2), U-Net and DenseNet variants, GIE/CBAM
variants, and STN/TPS components. These implementations are not referenced by
the standard `inference.py`, `eval.py`, or `train.py` execution paths. The
ResNet-101 implementation also contains an ImageNet weight download function,
but that function is disabled in the active constructor; standard inference
loads only the two local checkpoints described above.

## Task-aware ONNX directory processing

`process_onnx_tasks.py` reproduces all six inference modes with separate ONNX
Runtime sessions; it does not apply dewarping indiscriminately to every image.
The learned operations and deterministic prompt generation are arranged as
follows:

| Input naming convention | Selected task | ONNX sequence |
| --- | --- | --- |
| `for_appearance.*` | Appearance enhancement | Dynamic FP16 DocRes |
| `for_binarization.*` | Binarization | Dynamic FP16 DocRes |
| `for_debluring.*` or `for_deblurring.*` | Deblurring | Dynamic FP16 DocRes |
| `for_deshadowing.*` | Deshadowing | Dynamic FP16 DocRes |
| `for_dewarping.*` | Dewarping | MBD, then fixed FP32 DocRes |
| `for_end2end.*` or a numeric sample name | End-to-end | MBD and fixed FP32 DocRes, then dynamic FP16 DocRes twice |

End-to-end processing matches the Python pipeline order: dewarping, an
intermediate JPEG encode/decode, deshadowing, a second JPEG encode/decode, and
appearance enhancement. MBD receives `[1, 3, 448, 448]`, and the dewarping
DocRes pass receives `[1, 6, 256, 256]`, exactly matching `inference.py`.
Other tasks use a batch-1 FP16 model with dynamic height and width. Export it
from the checkpoint with opset 17 and automatic `onnxsim` validation:

```bash
uv run --frozen python export_docres_dynamic_onnx.py
```

`--opset` and `--output` override their defaults. `--disable-onnxsim` is the
only option that skips simplification. Process the complete `input/` directory
and write task-suffixed PNG files plus `outputs/manifest.json` with:

```bash
uv run --frozen python process_onnx_tasks.py --large-size 1120
```

The default `--large-size 1600` matches the PyTorch implementation and should
be used when sufficient GPU memory is available. On the NVIDIA GeForce RTX
3070 8 GiB used for the current test, 1600 x 1600 is out of memory. A standalone
1144 x 1144 DocRes pass succeeded, but it was not stable after the MBD and
fixed-DocRes sessions had run in the same process. The largest tested size that
completed all 11 task-assigned inputs consecutively was 1120 x 1120. Images
smaller than the Python pipeline's 1600-pixel threshold retain their padded
native resolution; for example, binarization ran at `[1, 6, 816, 1504]` and
deblurring at `[1, 6, 200, 200]`.

Every generated result retained its source height and width and passed the
finite-value and output-range checks. Comparisons with the available saved
PyTorch results were:

| Output | PSNR | SSIM | Mean absolute error |
| --- | ---: | ---: | ---: |
| `151_in_end2end.png` | 23.430 dB | 0.94372 | 7.830 |
| `189_origin_end2end.png` | 29.459 dB | 0.97430 | 4.216 |
| `208_in_end2end.png` | 30.504 dB | 0.92643 | 3.080 |
| `for_appearance_appearance.png` | 27.772 dB | 0.97975 | 4.719 |
| `for_binarization_binarization.png` | 51.320 dB | 0.99996 | 0.002 |
| `for_debluring_deblurring.png` | 62.590 dB | 0.99987 | 0.036 |
| `for_deshadowing_deshadowing.png` | 42.094 dB | 0.98767 | 1.398 |
| `for_dewarping_dewarping.png` | 40.771 dB | 0.98838 | 0.999 |
| `for_end2end_end2end.png` | 32.255 dB | 0.97481 | 3.560 |

The large-image differences are primarily explained by using 1120 instead of
the PyTorch pipeline's 1600 internal resolution. `190_in.png` and `218_in.png`
have no saved PyTorch references in this checkout, so only shape, range, and
visual checks were performed for those results. The four deterministic prompt
implementations were also compared array-for-array with `inference.py` and
matched exactly. For the same `[1, 6, 200, 200]` FP16 deblurring tensor, ONNX
and PyTorch model outputs had a mean absolute difference of `1.36e-4`, a
maximum difference of `0.00586`, and passed `rtol=0.01, atol=0.001`.
