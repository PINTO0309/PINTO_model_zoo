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

If you use the PINTO special model with the "fused_argmax_scale_ratio" option added, run the following script.

```
python demo_PIDNet_onnx_fused_argmax.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='pidnet_S_cityscapes_544x960_fused_argmax.onnx',
    )
```
