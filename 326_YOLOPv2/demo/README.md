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

## Demo
1. 384x640 - Heavy rain, dense fog and blurring - ONNX with Post-Process/NMS + TensorRT - yolopv2_post_384x640.onnx
    https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002161658_00000

  https://user-images.githubusercontent.com/33194443/188272391-df65a56f-3941-455b-86c7-59bc03f2d9c3.mp4

