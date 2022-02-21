# Demo projects

## SFace with ONNX Runtime in Python
```
python demo_SFace_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='face_recognition_sface_2021dec_112x112/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='112,112',
    )
```

## SFace with TensorFlow Lite in Python
```
python demo_SFace_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default=
        'face_recognition_sface_2021dec_112x112/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='112,112',
    )
```
