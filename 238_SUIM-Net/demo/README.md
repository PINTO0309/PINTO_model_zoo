# Demo projects

## SUIM-Net with ONNX Runtime in Python
```
python demo_SUIM-Net_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_rsb_480x640/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='480,640',
    )
```

## SUIM-Net with TensorFlow Lite in Python
```
python demo_SUIM-Net_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_rsb_480x640/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='480,640',
    )
```


