# Demo projects

## Diffusion-Low-Light with ONNX Runtime in Python
```
python demo_Diffusion-Low-Light_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='diffusion_low_light_1x3x192x320.onnx',
    )
```
