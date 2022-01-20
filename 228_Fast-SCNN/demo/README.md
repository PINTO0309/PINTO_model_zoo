# Demo projects

## Fast-SCNN with ONNX Runtime in Python
```
python demo_Fast-SCNN_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='fast_scnn_768x1344/fast_scnn_768x1344.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='768,1344',
    )
```

## Fast-SCNN with TensorFlow Lite in Python
```
python demo_Fast-SCNN_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='fast_scnn_768x1344/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='768,1344',
    )
```

## Fast-SCNN with TensorRT in Python
- https://github.com/NobuoTsukamoto/tensorrt-examples/blob/main/python/fast_scnn
