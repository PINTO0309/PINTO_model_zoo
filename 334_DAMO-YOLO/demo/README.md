# Demo projects

## DAMO-YOLO with ONNX Runtime in Python
```
python demo_DAMO-YOLO_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='damoyolo_tinynasL20_T_192x320.onnx',
    )
```
<br>
A model with post-processing runs like this.<br>

```
python demo_DAMO-YOLO_with_postprocess_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='damoyolo_tinynasL20_T_192x320_post.onnx',
    )
```

- TensorRT + `demo_DAMO-YOLO_with_postprocess_onnx.py` + `damoyolo_tinynasL35_M_384x640_post.onnx`

    https://user-images.githubusercontent.com/33194443/206323392-28ac9707-8c12-4431-870b-a90399955b83.mp4

