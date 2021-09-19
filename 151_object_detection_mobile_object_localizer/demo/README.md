# Demo projects

## Mobile Object Localizer with ONNX Runtime in Python
```
python demo_onnx
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model/model_float32.onnx',
    )
    parser.add_argument("--score", type=float, default=0.2)
```

## Mobile Object Localizer with TensorFlow Lite in Python
```
python demo_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model/model_float16_quant.tflite',
    )
    parser.add_argument("--score", type=float, default=0.2)
```

