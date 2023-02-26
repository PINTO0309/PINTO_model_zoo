# Demo projects

## DEA-Net with ONNX Runtime in Python
```
python demo_DEA-Net_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        '-d',
        '--device',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-f',
        '--movie_file',
        type=str,
        default='',
    )
    parser.add_argument(
        "--model",
        type=str,
        default='dea_net_haze4k_360x640.onnx',
    )
```

https://user-images.githubusercontent.com/33194443/221406178-c5daa6d4-de7f-464c-8fad-98763192cf90.mp4

