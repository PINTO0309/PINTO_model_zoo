# Demo projects

## E2Pose with ONNX Runtime in Python
```
python demo_E2Pose_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model/e2epose_1x3x512x512_post.onnx',
    )
```
