# Demo projects

## MSBDN-DFF with ONNX Runtime in Python
```
python demo_MSBDN-DFF_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='msbdn_dff_192x320/msbdn_dff_192x320.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='192,320',
    )
```

## MSBDN-DFF with TensorFlow Lite in Python
```
python demo_MSBDN-DFF_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='msbdn_dff_192x320/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='192,320',
    )
```


