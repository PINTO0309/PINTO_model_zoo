# Demo projects

## U-2-Net with TensorRT in Python
https://github.com/NobuoTsukamoto/tensorrt-examples/blob/main/python/u2net/README.md

## U-2-Net with ONNX Runtime in Python
```
demo_u2net_onnx.py
  or
demo_u2net_portrait_onnx.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='../01_float32/u2netp_320x320.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='320,320',
    )
```

## U-2-Net with TensorFlow Lite in Python
```
demo_u2net_tflite.py
  or
demo_u2net_portrait_tflite.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.
```python
    parser.add_argument(
        "--model",
        type=str,
        default=
        '../05_float16_quantization/u2netp_320x320_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='320,320',
    )
```
