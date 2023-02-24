# Demo projects

## DGNet with ONNX Runtime in Python
```
python demo_DGNet_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='dgnet_s_480x640/dgnet_s_480x640.onnx',
    )
```
