# Demo projects

## HINet with ONNX Runtime in Python
```
python demo_hinet_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default=
        'hinet_derain_test100_256x320/hinet_derain_test100_256x320.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,320',
    )
```

## HINet with TensorFlow Lite in Python
```
python demo_hinet_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='hinet_derain_test100_256x320/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,320',
    )
```


