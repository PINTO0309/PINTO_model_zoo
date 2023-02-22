# Demo projects

## PyDNet2 with ONNX Runtime in Python
```
python demo_PyDNet2_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='pydnet2_192x512/model_float32.onnx',
    )
```

## PyDNet2 with TensorFlow Lite in Python
```
python demo_PyDNet2_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='pydnet2_192x512/model_float16_quant.tflite',
    )
```


