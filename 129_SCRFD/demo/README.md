# Demo projects

## SCRFD with ONNX Runtime in Python
I have confirmed the operation only on some models.<br>
Some models may not work.
```
python demo_SCRFD_onnx.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.
```python
    parser.add_argument(
        "--model",
        type=str,
        default=
        'saved_model_scrfd_500m_bnkps_480x640/scrfd_500m_bnkps_480x640.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='480,640',
    )
```

## SCRFD with TensorFlow Lite in Python
I have confirmed the operation only on some models.<br>
Some models may not work.
```
python demo_SCRFD_tflite.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.
```python
    parser.add_argument(
        "--model",
        type=str,
        default=
        'saved_model_scrfd_500m_bnkps_480x640/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='480,640',
    )
```
