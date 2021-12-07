# Demo projects

## AGLLNet with ONNX Runtime in Python
```
python demo_AGLLNet_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_256x256/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,256',
    )
```

## AGLLNet with TensorFlow Lite in Python
```
python demo_AGLLNet_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_256x256/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,256',
    )
```


