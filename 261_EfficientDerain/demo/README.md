# Demo projects

## EfficientDerain with ONNX Runtime in Python
```
python demo_EfficientDerain_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='efficientderain_v4_spa_320x480.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='320,480',
    )
```
