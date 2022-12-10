# Demo projects

## PIDNet with ONNX Runtime in Python
```
python demo_PIDNet_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='pidnet_S_cityscapes_544x960.onnx',
    )
```
