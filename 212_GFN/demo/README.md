# Demo projects

## GFN with ONNX Runtime in Python
```
python demo_GFN_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='gfn_128x128/gfn_128x128.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='128,128',
    )
```

## GFN with TensorFlow Lite in Python
```
python demo_GFN_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='gfn_128x128/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='128,128',
    )
```


