# Demo projects

## PPHumanSeg with ONNX Runtime in Python
```
python demo_pphumanseg_onnx.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_human_segmentation_pphumanseg/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='192,192',
    )
```

## PPHumanSeg with TensorFlow Lite in Python
```
python demo_pphumanseg_tflite.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.
```python
    parser.add_argument(
        "--model",
        type=str,
        default=
        'saved_model_human_segmentation_pphumanseg/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='192,192',
    )
```
