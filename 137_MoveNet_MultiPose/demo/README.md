# Demo projects

## Pose estimation (movenet/multipose/lightning) with TensorFlow Lite in C++
https://github.com/iwatake2222/play_with_tflite/tree/master/pj_tflite_pose_movenet_multi

## Pose estimation (movenet/multipose/lightning) with ONNX Runtime in Python
```
python demo_multipose_onnx
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_192x256/model_float32.onnx',
    )
    parser.add_argument("--input_size", type=str, default='192,256')
```

## Pose estimation (movenet/multipose/lightning) with TensorFlow Lite in Python
```
python demo_multipose_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_192x256/model_float16_quant.tflite',
    )
    parser.add_argument("--input_size", type=str, default='192,256')
```

## 6 persons version
https://github.com/Kazuhito00/MoveNet-Python-Example

## 10 persons version (MoveNet model optimized)
https://github.com/PINTO0309/MoveNet-Python-Example
