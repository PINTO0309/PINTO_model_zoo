# Demo projects

## Zero-DCE-improved with ONNX Runtime in Python
```
python demo_Zero-DCE-improved_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='zero_dce_improved_180x320/zero_dce_improved_180x320.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )
```

## Zero-DCE-improved with TensorFlow Lite in Python
```
python demo_Zero-DCE-improved_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='zero_dce_improved_180x320/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )
```


