# Demo projects

## Real-CUGAN with ONNX Runtime in Python
```
python demo_Real-CUGAN_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default=
        'realcugan_2x_tile3_denoise_240x320/realcugan_2x_tile3_denoise_240x320.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='240,320',
    )
```
