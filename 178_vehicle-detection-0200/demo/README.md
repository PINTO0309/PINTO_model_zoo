# Demo projects

## Vehicle Detection with OpenCV in Rust
https://github.com/iwatake2222/opencv_sample_in_rust/tree/master/pj_dnn_det_vehicle-detection-0200

## Vehicle Detection with ONNX Runtime in Python
```
python demo_vehicle-detection-0200_onnx.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_256x256/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,256',
    )
```

## Vehicle Detection with TensorFlow Lite in Python
```
python demo_vehicle-detection-0200_tflite.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_256x256/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,256',
    )
```
