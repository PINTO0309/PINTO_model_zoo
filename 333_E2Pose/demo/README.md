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
        default='saved_model/e2epose_resnet50_1x3x512x512.onnx',
    )
```

- CUDA - e2epose_resnet101_1x3x512x512.onnx
    
    https://user-images.githubusercontent.com/33194443/206191133-5aac3b34-d74f-403d-9ce9-90cc0bccc96e.mp4


- TensorRT - e2epose_resnet101_1x3x512x512.onnx
    
    https://user-images.githubusercontent.com/33194443/206191168-3a651c20-7e50-4920-9ca9-8b77d678cbda.mp4

