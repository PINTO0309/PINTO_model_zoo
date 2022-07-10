# Demo projects

## FastestDet with ONNX Runtime in Python
```
python demo_FastestDet_with_postprocess_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='fastestdet_post_352x352.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='352,352',
    )
```
