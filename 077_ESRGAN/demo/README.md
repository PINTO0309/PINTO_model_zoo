# Demo projects

## ESRGAN with TensorRT in Python
https://github.com/NobuoTsukamoto/tensorrt-examples/blob/main/python/esrgan/README.md

## ESRGAN with TensorFlow Lite in Python
https://github.com/NobuoTsukamoto/tflite-cv-example/blob/master/super_resolution/README.md

## ESRGAN with ONNX Runtime in Python
```
python demo_ESRGAN_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_50x50/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='50,50',
    )
```