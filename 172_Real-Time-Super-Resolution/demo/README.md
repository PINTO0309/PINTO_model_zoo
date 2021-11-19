# Demo projects

## Real-Time-Super-Resolution with ONNX Runtime in Python
```
python demo_Real-Time-Super-Resolution_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_96x96/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='96,96',
    )
```

## Real-Time-Super-Resolution with TensorFlow Lite in Python
```
python demo_Real-Time-Super-Resolution_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_96x96/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='96,96',
    )
```


