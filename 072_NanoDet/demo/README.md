# Demo projects

## NanoDet with OpenCV(cv::dnn + ONNX) in C++
https://github.com/iwatake2222/play_with_tflite/tree/master/pj_tflite_det_nanodet

## NanoDet with ONNX Runtime in Python
```
python demo_onnx.py
```

If you want to change the model, change the NanoDetONNX argument.
```python
if __name__ == '__main__':
    # Initialize NanoDetTFlite Class
    nanodet = NanoDetONNX(
        model_path='saved_model_nanodet_m_320x320/nanodet_m_320x320.onnx',
        input_shape=320,
    )
```

## NanoDet with TensorFlow Lite in Python
```
python demo_tflite.py
```

If you want to change the model, change the NanoDetTFLite argument.
```python
if __name__ == '__main__':
    # Initialize NanoDetTFlite Class
    nanodet = NanoDetTFLite(
        model_path='saved_model_nanodet_320x320/model_float16_quant.tflite',
        input_shape=320,
    )
```

