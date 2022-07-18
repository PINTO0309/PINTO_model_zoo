# Demo projects

## FastestDet with ONNX Runtime in Python
```
python demo_FastestDet_with_postprocess_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='fastestdet_post_352x352.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='352,352',
    )
```

## FastestDet with TensorFlow Lite in C++
https://github.com/iwatake2222/play_with_tflite/tree/master/pj_tflite_det_fastestdet
