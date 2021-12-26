# Demo projects

## Zero-DCE-TF with ONNX Runtime in Python
```
python demo_Zero-DCE-TF_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_180x320/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )
```

## Zero-DCE-TF with TensorFlow Lite in Python
```
python demo_Zero-DCE-TF_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_180x320/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )
```


