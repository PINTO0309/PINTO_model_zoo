# Demo projects

## TransWeather with ONNX Runtime in Python
```
python demo_TransWeather_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='transweather_736x1280/transweather_736x1280.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='736,1280',
    )
```
