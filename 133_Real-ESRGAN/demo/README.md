# Demo projects

## Real-ESRGAN with ONNX Runtime in Python
```
python demo_Real-ESRGAN_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_64x64/realesrgan_64x64.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='64,64',
    )
```

## Real-ESRGAN with TensorFlow Lite in Python
```
python demo_Real-ESRGAN_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_64x64/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='64,64',
    )
```


