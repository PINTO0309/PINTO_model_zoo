# Demo projects

## ImageForensicsOSN with ONNX Runtime in Python
```
python demo_ImageForensicsOSN_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='imageforensicsosn_480x640/imageforensicsosn_480x640.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='480,640',
    )
```
