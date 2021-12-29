# Demo projects

## RUAS with ONNX Runtime in Python
```
python demo_RUAS_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='lol_180x320/lol_180x320.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default='lol',
        choices=['lol', 'upe', 'dark'],
    )
```

## RUAS with TensorFlow Lite in Python
```
python demo_RUAS_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='lol_180x320/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default='lol',
        choices=['lol', 'upe', 'dark'],
    )
```


