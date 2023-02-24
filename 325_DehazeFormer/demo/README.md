# Demo projects

## DehazeFormer with ONNX Runtime in Python
```
python demo_DehazeFormer_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='dehazeformer_t_outdoor_360x640.onnx',
    )
```
