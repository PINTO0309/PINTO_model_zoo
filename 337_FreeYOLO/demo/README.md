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
<br>
A model with post-processing runs like this.<br>

```
python demo_FreeYOLO_with_postprocess_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='yolo_free_nano_640x640_post.onnx',
    )
```
