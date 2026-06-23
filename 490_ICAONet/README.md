# 490_ICAONet

ICAONet is the first deep multitask learning-based method designed for automatic evaluation of the photographic requirements of the ISO/IEC 19794-5 standard.

# Running ONNX/TFLite Pipeline

`predict_onnx_tflite.py` runs a full ONNX/TFLite inference pipeline without the
TensorFlow/Keras `h5` model. It detects only classid `7` (`Head`) with
`resources/models/yolomit_t_wholebody28_1x3x480x640.onnx`, square-crops the
detected head without the `1.5x` expansion used by `predict.py`, resizes the
crop to `160x160`, saves it as `data/test_160x160.png`, and runs ICAONet.

```bash
python predict_onnx_tflite.py
```

The default command is equivalent to:

```bash
python predict_onnx_tflite.py \
--image data/test.png \
--detector resources/models/yolomit_t_wholebody28_1x3x480x640.onnx \
--backend onnx \
--icaonet-onnx resources/models/icaonet_1x3x160x160.onnx \
--icaonet-tflite resources/models/saved_model_icaonet_1x3x160x160/icaonet_1x3x160x160_float32.tflite \
--class-id 7 \
--score-threshold 0.25 \
--iou-threshold 0.45 \
--crop-output data/test_160x160.png
```

Choose the ICAONet runtime with `--backend`:

```bash
python predict_onnx_tflite.py --backend onnx
python predict_onnx_tflite.py --backend tflite
python predict_onnx_tflite.py --backend both
```

`--backend both` prints ONNX and TFLite `output_reqs` and their
`max_abs_diff`. The TFLite path uses `ai_edge_litert.Interpreter` first and
falls back to `tflite_runtime.Interpreter`; it does not import TensorFlow.

The script prints the selected Head detection score, the detected `xyxy` box,
the square crop box, the saved `160x160` crop path, and each ICAONet
`output_reqs` score with its ICAO requirement id and description. Use
`--crop-output ""` to skip writing the crop image.

The `output_reqs` values are sigmoid scores in the `0..1` range. In this
repository's helper API, a higher value means the image is judged more
compliant for that requirement, not that the problem named by the label is more
severe. With the default threshold convention, `score >= 0.5` is treated as
compliant and `score < 0.5` is treated as non-compliant. If you want a
problem-oriented score for labels such as `Blurred`, `Pixelation`, or
`Varied background`, read it as approximately `1.0 - score`.

```
python predict_onnx_tflite.py --backend onnx

Selected Head detection
score: 0.946571
xyxy: [296.45391845703125, 147.14996337890625, 437.19793701171875, 335.9950866699219]
square_xyxy: [272.40338134765625, 147.14996337890625, 461.24847412109375, 335.9950866699219]

crop_output: data/test_160x160.png

Backend: onnx
[02] Blurred: 0.20035028457641602
[03] Looking away: 0.9776184558868408
[04] Ink marked/creased: 0.9997095465660095
[05] Unnatural skin tone: 0.504340648651123
[06] Too dark/light: 0.9961643218994141
[07] Washed out: 0.9980594515800476
[08] Pixelation: 0.9310570955276489
[09] Hair across eyes: 0.9962046146392822
[10] Eyes closed: 0.9938639402389526
[11] Varied background: 0.014305025339126587
[12] Roll/pitch/yaw rotations greater than a predefined thresholds: 0.7484405040740967
[13] Flash reflection on skin: 0.623089611530304
[14] Red eyes: 0.9961295127868652
[15] Shadows behind head: 0.2613902986049652
[16] Shadows across face: 0.7242169380187988
[17] Dark tinted lenses: 0.9999758005142212
[18] Flash reflection on lenses: 0.9811655282974243
[19] Frames too heavy: 0.9974024295806885
[20] Frame covering eyes: 0.7655755281448364
[21] Hat/cap: 0.0012105703353881836
[22] Veil over face: 0.9999406337738037
[23] Mouth open: 0.9977437257766724
[24] Presence of other faces or toys too close to face: 0.9756643176078796
```

# Export TensorFlow 2 SavedModels

`export_saved_model.py` exports four TensorFlow 2 SavedModels from
`resources/models/icaonet_with_decoder.h5`. By default, the exported models
accept RGB input and swap channels to BGR at the model boundary because ICAONet
was trained with OpenCV BGR images.

```sh
python export_saved_model.py
```

Default RGB-input outputs:

| Directory | Outputs |
| --- | --- |
| `resources/models/saved_model_icaonet_rgb_all_branches` | `decoded`, `output_reqs`, `output_eyes`, `output_pixelation` |
| `resources/models/saved_model_icaonet_rgb_without_decoder` | `output_reqs`, `output_eyes`, `output_pixelation` |
| `resources/models/saved_model_icaonet_rgb_without_decoder_eyes` | `output_reqs`, `output_pixelation` |
| `resources/models/saved_model_icaonet_rgb_output_reqs` | `output_reqs` |

Each SavedModel has two signatures:

| Signature | Input |
| --- | --- |
| `serving_default` | `image_rgb` float32, shape `(batch, 160, 160, 3)`, RGB, normalized to `0..1` |
| `from_uint8` | `image_rgb_uint8` uint8, shape `(batch, 160, 160, 3)`, RGB |

To preserve the original OpenCV BGR input contract, export with:

```sh
python export_saved_model.py --input-color-order bgr
```

# Model outputs

The current model files under `resources/models` are not the same single-output
classification model shown in the original `v0.6.4` training notebook. They are
merged multitask models with separate heads for ICAO requirements, eye
coordinates, and pixelation.

All model inputs are cropped face images resized to `160x160`, represented as
float32 values normalized to `0..1`. The notebooks use OpenCV, so the in-memory
image channel order is BGR unless explicitly converted for display.

## `icaonet.h5`

`resources/models/icaonet.h5` has one input and three sigmoid outputs:

```py
y_reqs, y_eyes, y_pixelation = model.predict(image)
```

| Output | Shape | Meaning |
| --- | --- | --- |
| `output_reqs` | `(batch, 23)` | Scores for the 23 ISO/ICAO photographic requirements. |
| `output_eyes` | `(batch, 4)` | Normalized eye coordinates: `[x_left, y_left, x_right, y_right]`. |
| `output_pixelation` | `(batch, 1)` | Dedicated pixelation score from a separate pixelation branch. |

`output_reqs` follows the order used by `src.iso_standard.PhotographicRequirements`:

| Index | Requirement |
| --- | --- |
| 0 | `blurred` |
| 1 | `looking_away` |
| 2 | `ink_marked_creased` |
| 3 | `unnatural_skin_tone` |
| 4 | `too_dark_light` |
| 5 | `washed_out` |
| 6 | `pixelation` |
| 7 | `hair_across_eyes` |
| 8 | `eyes_closed` |
| 9 | `varied_background` |
| 10 | `roll_pitch_yaw` |
| 11 | `flash_reflection_on_skin` |
| 12 | `red_eyes` |
| 13 | `shadows_behind_head` |
| 14 | `shadows_across_face` |
| 15 | `dark_tinted_lenses` |
| 16 | `flash_reflection_on_lenses` |
| 17 | `frames_too_heavy` |
| 18 | `frame_covering_eyes` |
| 19 | `hat_cap` |
| 20 | `veil_over_face` |
| 21 | `mouth_open` |
| 22 | `presence_of_other_faces_or_toys` |

The requirement scores can be wrapped with the helper class:

```py
from src.iso_standard import PhotographicRequirements

reqs = PhotographicRequirements(*y_reqs.squeeze())
print(reqs.blurred.value)
print(reqs.blurred.is_compliant(threshold=0.5))
```

`Requirement.is_compliant(threshold)` returns `True` when the score is greater
than or equal to the threshold. Therefore high `output_reqs` values should be
read as "passes this requirement" under the repository convention. For example,
`[08] Pixelation: 0.93` means the image is likely compliant with respect to
pixelation, while `[11] Varied background: 0.01` means it is likely
non-compliant for the background requirement. These scores are model outputs,
not calibrated probabilities.

`output_eyes` is normalized by the input size. Convert it back to pixel
coordinates by multiplying by `160`:

```py
x_left, y_left, x_right, y_right = y_eyes[0] * 160
```

`output_reqs[:, 6]` and `output_pixelation[:, 0]` are both pixelation-related
scores. The former is the pixelation score inside the 23-requirement head, while
the latter is produced by the dedicated pixelation branch that was merged into
the current model.

## `icaonet.pb`

`resources/models/icaonet.pb` is the frozen TensorFlow graph version of
`icaonet.h5`. It exposes the same three outputs.

| Tensor | Shape |
| --- | --- |
| `input:0` | `(batch, 160, 160, 3)` |
| `output_reqs/Sigmoid:0` | `(batch, 23)` |
| `output_eyes/Sigmoid:0` | `(batch, 4)` |
| `output_pixelation/Sigmoid:0` | `(batch, 1)` |

## `icaonet_with_decoder.h5`

`resources/models/icaonet_with_decoder.h5` keeps the autoencoder decoder output
in addition to the three inference heads:

```py
decoded, y_reqs, y_eyes, y_pixelation = model.predict(image)
```

| Output | Shape | Meaning |
| --- | --- | --- |
| `decoded` | `(batch, 160, 160, 3)` | Reconstructed normalized image. |
| `output_reqs` | `(batch, 23)` | Scores for the 23 ISO/ICAO photographic requirements. |
| `output_eyes` | `(batch, 4)` | Normalized eye coordinates. |
| `output_pixelation` | `(batch, 1)` | Dedicated pixelation score. |

For visualization, convert the reconstructed BGR image to RGB in the same way as
the inference notebook:

```py
plt.imshow(decoded[0, :, :, ::-1])
```

## Network summary

The merged model uses a shared convolutional encoder:

```text
input 160x160x3
 -> Conv2D 32 + BatchNorm + ReLU + MaxPool
 -> Conv2D 64 + BatchNorm + ReLU + MaxPool
 -> Conv2D 128 + BatchNorm + ReLU + MaxPool
 -> Conv2D 256 + BatchNorm + ReLU + MaxPool
 -> Conv2D 256 + BatchNorm + tanh = encoded 10x10x256
```

The heads are:

```text
output_reqs:
encoded -> GlobalAveragePooling -> Dropout -> Dense64 -> Dropout -> Dense32
        -> Dense23 sigmoid

output_eyes:
encoded -> GlobalAveragePooling -> Dropout -> Dense64 -> Dropout -> Dense64
        -> Dense4 sigmoid

output_pixelation:
pool_2 -> GlobalAveragePooling -> Dropout -> Dense128 -> Dropout -> Dense128
       -> Dropout -> Dense128 -> Dense1 sigmoid
```

`icaonet_with_decoder.h5` also includes:

```text
decoded:
encoded -> Conv2DTranspose256 -> Conv2DTranspose128 -> Conv2DTranspose64
        -> Conv2DTranspose32 -> Conv2D3 sigmoid
```

## Cited
  I am very grateful for their excellent work.
  - ICAONet

    https://github.com/arnaldog12/icaonet

    ```bibtex
    @article{DEANDRADEESILVA2022116756,
        title = {A collaborative deep multitask learning network for face image compliance to ISO/IEC 19794-5 standard},
        journal = {Expert Systems with Applications},
        pages = {116756},
        year = {2022},
        issn = {0957-4174},
        doi = {https://doi.org/10.1016/j.eswa.2022.116756},
        url = {https://www.sciencedirect.com/science/article/pii/S0957417422002226},
        author = {Arnaldo Gualberto {de Andrade e Silva} and Herman Martins Gomes and Leonardo Vidal Batista},
        keywords = {Face quality, ICAO, ISO/IEC 19794-5, Multitask Learning, Autoencoders, Deep Learning},
        abstract = {The face is considered the primary biometric trait for machine-readable travel documents, like passports. In this context, the ISO/IEC 19794-5 standard defines a set of photographic requirements to ensure image quality and simplify the face recognition process. However, the assessment of face image compliance to the ISO/ICAO standard is still mostly performed by humans today due to the lack of automatic evaluation systems to perform this task. In this paper, we present the first deep multitask learning-based method designed for automatic evaluation of the photographic requirements of the ISO/IEC 19794-5 standard, called ICAONet. We extended undercomplete Autoencoders to employ a multi-and-collaborative learning approach, where both supervised and unsupervised learning is performed concurrently and in a collaborative manner. The method is trained using an ad hoc image dataset and evaluated by an official benchmark system also used by other approaches presented in the literature. The results show that our method achieves the best results in terms of Equal Error Rate for 9 out of the 23 photographic requirements of ISO/IEC 19794-5, which was not achieved by any other individual method evaluated. Therefore, the proposed method can be considered the best overall solution among academic works published in the literature and private SDKs. Overall, the median Equal Error Rate (3.3%) is also competitive. Finally, in terms of running time, the proposed method stands out among the fastest to evaluate all 23 requirements according to the official benchmark.}
    }
    ```
