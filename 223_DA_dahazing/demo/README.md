# Demo projects

## DA_dehazing with ONNX Runtime in Python
```
python demo_DA_dehazing_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='sdehazingnet_192x320/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='192,320',
    )
```

## DA_dehazing with TensorFlow Lite in Python
```
python demo_DA_dehazing_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='sdehazingnet_192x320/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='192,320',
    )
```


