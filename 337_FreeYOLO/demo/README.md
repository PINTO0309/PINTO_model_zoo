# Demo projects

## FreeYOLO with ONNX Runtime in Python
```
python demo_FreeYOLO_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='yolo_free_nano_192x320.onnx',
    )
```
