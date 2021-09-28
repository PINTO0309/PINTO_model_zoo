# Demo projects

## FastDepth with TensorRT in Python
https://github.com/NobuoTsukamoto/tensorrt-examples/tree/main/python/fast_depth

## FastDepth with ONNX Runtime in Python
```
python demo_fast_depth_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='fast_depth_128x160.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='128,160',
    )
```

## FastDepth with TensorFlow Lite in Python
```
python demo_fast_depth_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='128,160',
    )
```


