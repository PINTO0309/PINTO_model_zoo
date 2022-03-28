# Demo projects

## IMDN with ONNX Runtime in Python
```
python demo_IMDN_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='imdn_128x128/imdn_128x128.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='128,128',
    )
```

