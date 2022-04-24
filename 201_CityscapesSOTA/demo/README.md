# Demo Projects

## Segmentation using PaddleSeg CityscapesSOTA with TensorRT in C++
https://github.com/iwatake2222/play_with_tensorrt/tree/master/pj_tensorrt_seg_paddleseg_cityscapessota

## InferenceHelper Sample ROS
- https://github.com/iwatake2222/InferenceHelper_Sample_ROS
- Using this model with TensorFlow Lite / ONNX Runtime in ROS2 (rclcpp)

## Segmentation using PaddleSeg CityscapesSOTA with ONNX Runtime in Python
```
python demo_CityscapesSOTA_onnx.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_180x320/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )
```

## Segmentation using PaddleSeg CityscapesSOTA with TensorFlow Lite in Python
```
python demo_CityscapesSOTA_tflite.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_180x320/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )
```

