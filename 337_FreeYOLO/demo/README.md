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

## FreeYOLO with ONNX Runtime in JavaScript(Typescript)

https://w-okada.github.io/free-yolo-pinto-onnx-test/

[movie](https://twitter.com/DannadoriYellow/status/1605131504795123712)

[movie](https://twitter.com/DannadoriYellow/status/1605131750522654720)

