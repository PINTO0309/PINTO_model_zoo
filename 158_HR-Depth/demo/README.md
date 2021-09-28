# Demo projects

## HR-Depth with ONNX Runtime in Python
```
python demo_hr_depth_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='lite_hr_depth_k_t_encoder_depth_192x640.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='192,640',
    )
```

## HR-Depth with TensorFlow Lite in Python
```
python demo_hr_depth_tflite.py
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
        type=str,
        default='192,640',
    )
```


