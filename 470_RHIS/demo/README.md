# Demo projects

## RHIS with ONNX Runtime in Python
Download the weights and run the following script.
```
python demo_rhis.py
```
If you want to change the model, specify it with an argument.
```python
parser.add_argument(
    "--model",
    type=str,
    default="best_model_b1_640x640_80x60_0.8551_dil1.onnx",
    help="Path to the ONNX model file.",
)
```
<br>
In this script, the regions of interest (ROIs) are hard-coded in the source code as examples.<br>
When you integrate it into your system, it is recommended to specify the bounding box coordinates obtained from any object detection model as the ROIs.

```python
image_path = "sample.jpg"
rois_unnormalized = np.array(
    [
        [190, 0, 626, 533],
        [183, 239, 679, 497],
        [478, 0, 800, 532],
    ],
    dtype=np.float32,
)
```

<img width="800" height="533" alt="RHIS Demo: masks_screenshot_06 09 2025" src="https://github.com/user-attachments/assets/d3feedba-3287-4106-85e7-310fd15ac4dc" />

## About sample images
The sample image uses the image of "[PAKUTASO](https://www.pakutaso.com/)".<br>
If you want to use the image itself for another purpose, you must follow the [userpolicy of PAKUTASO](https://www.pakutaso.com/userpolicy.html).
