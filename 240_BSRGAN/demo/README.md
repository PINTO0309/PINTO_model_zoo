# Demo projects

## BSRGAN with ONNX Runtime in Python
```
python demo_BSRGAN_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='bsrganx2_128x128/bsrganx2_128x128.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='128,128',
    )
```

## BSRGAN with TensorFlow Lite in Python
```
python demo_BSRGAN_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='bsrganx2_128x128/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='128,128',
    )
```


