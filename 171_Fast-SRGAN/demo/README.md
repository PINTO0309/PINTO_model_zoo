# Demo projects

## Fast-SRGAN with ONNX Runtime in Python
```
python demo_fast_srgan_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='model_128x128/model_128x128.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='128,128',
    )
```

## Fast-SRGAN with TensorFlow Lite in Python
```
python demo_fast_srgan_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='model_128x128/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='128,128',
    )
```


