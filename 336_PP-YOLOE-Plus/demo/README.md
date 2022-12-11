# Demo projects

## PP-YOLOE-Plus with ONNX Runtime in Python
```
python demo_PP-YOLOE-Plus_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='ppyoloe_plus_crn_s_80e_coco_640x640.onnx',
    )
```
