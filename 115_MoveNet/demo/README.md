# Demo projects

## Pose estimation (MoveNet/SinglePose/Lightning/v4) with TensorFlow Lite in C++
https://github.com/iwatake2222/play_with_tflite/tree/master/pj_tflite_pose_movenet

## Pose estimation (MoveNet/SinglePose/Lightning/v4 or Thunder/v4) with ONNX Runtime in Python
```
python demo_singlepose_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=192,
        choices=[192, 256],
    )
```

## Pose estimation (MoveNet/SinglePose/Lightning/v4 or Thunder/v4) with TensorFlow Lite in Python
```
python demo_singlepose_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=192,
        choices=[192, 256],
    )
```

## 6 persons version
https://github.com/Kazuhito00/MoveNet-Python-Example

## 10 persons version (MoveNet model optimized)
https://github.com/PINTO0309/MoveNet-Python-Example
