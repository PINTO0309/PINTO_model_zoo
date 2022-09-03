# Demo Projects

## YOLOPv2 with TensorRT in C++
https://github.com/iwatake2222/play_with_tensorrt/tree/master/pj_tensorrt_perception_yolopv2

## YOLOPv2 with MNN in C++ (and Android)
https://github.com/iwatake2222/play_with_mnn/tree/master/pj_mnn_perception_yolopv2

## YOLOPv2(yolopv2_post) with ONNX Runtime in Python
```
python demo_yolopv2_post_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='yolopv2_post_192x320.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='192,320',
    )
```
