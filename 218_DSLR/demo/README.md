# Demo projects

## DSLR with ONNX Runtime in Python
```
python demo_DSLR_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='dslr_256x256/dslr_256x256.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,256',
    )
```

## DSLR with TensorFlow Lite in Python
```
python demo_DSLR_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='dslr_256x256/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,256',
    )
```


