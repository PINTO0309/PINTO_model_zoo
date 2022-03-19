# Demo projects

## FD-GAN with ONNX Runtime in Python
```
python demo_FD-GAN_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='fdgan_real_192x320.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='192,320',
    )
```
