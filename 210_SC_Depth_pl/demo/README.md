# Demo projects

## SC Depth pl with ONNX Runtime in Python
```
python demo_SC_Depth_pl_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='sc_depth_bonn_scv3_192x320.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='192,320',
    )
```
