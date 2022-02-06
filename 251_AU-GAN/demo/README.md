# Demo projects

## AU-GAN with ONNX Runtime in Python
```
python demo_AU-GAN_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_480x640/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='480,640',
    )
```

## AU-GAN with TensorFlow Lite in Python
```
python demo_AU-GAN_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_480x640/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='480,640',
    )
```


