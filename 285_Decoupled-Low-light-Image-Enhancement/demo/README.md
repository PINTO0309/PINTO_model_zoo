# Demo projects

## Decoupled-Low-light-Image-Enhancement with ONNX Runtime in Python
```
python demo_Decoupled-LLIE_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='onnx_net_merged_180x320/model_float32_final.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )
```

