# Demo projects

## Road Segmentation Adas with TensorFlow Lite in C++
https://github.com/iwatake2222/play_with_tflite/tree/master/pj_tflite_ss_road-segmentation-adas-0001

## Self-Driving-ish Computer Vision System
- https://github.com/iwatake2222/self-driving-ish_computer_vision_system
- Using this model with TensorRT / TensorFlow Lite in C++

## InferenceHelper Sample ROS
- https://github.com/iwatake2222/InferenceHelper_Sample_ROS
- Using this model with TensorFlow Lite / ONNX Runtime in ROS2 (rclcpp)

## Road Segmentation Adas with TensorFlow Lite in Python
```
python demo_road-segmentation-adas-0001_tflite.py
```

If you want to change the model, specify it with an argument.<br>
If you want to enter a video instead of a webcam, specify the "movie" argument.
```python
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='512,896',
    )
```

## Road Segmentation Adas with ONNX Runtime in Python
```
python demo_road-segmentation-adas-0001_onnx.py
```

If you want to change the model, specify it with an argument.<br>
If you want to enter a video instead of a webcam, specify the "movie" argument.
```python
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='512,896',
    )
```
