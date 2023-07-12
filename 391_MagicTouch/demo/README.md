# Demo projects

## MagicTouch with ONNX Runtime in Python
```
python demo_MagicTouch_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument("--image", type=str, default='sample.jpg')
    parser.add_argument(
        "--model",
        type=str,
        default='magic_touch.onnx',
    )
```

## MagicTouch with TensorFlow Lite in Python
```
python demo_MagicTouch_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument("--image", type=str, default='sample.jpg')
    parser.add_argument(
        "--model",
        type=str,
        default='magic_touch_float32.tflite',
    )
```

---
https://github.com/PINTO0309/PINTO_model_zoo/assets/37477845/6198b03c-8327-496c-a414-beb42bd39253
