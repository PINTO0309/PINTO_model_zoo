# Demo projects

## FSRE-Depth with ONNX Runtime in Python
```
python demo_FSRE-Depth_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='fsre_depth_192x320/fsre_depth_full_192x320.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='192,320',
    )
```
