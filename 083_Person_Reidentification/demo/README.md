# Demo projects

## YOLOX-Nano + Deep SORT (using person reidentification model) with TensorFlow Lite in C++
https://github.com/iwatake2222/play_with_tflite/tree/master/pj_tflite_track_deepsort_person-reidentification

## Person_Reidentification with ONNX Runtime in Python
```
python demo_Person_Reidentification_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='resources/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,128',
    )
```

## Person_Reidentification with TensorFlow Lite in Python
```
python demo_Person_Reidentification_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='resources/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,128',
    )
```
