# Demo projects

## DRBL Stage1 with ONNX Runtime in Python
```
python demo_DRBL_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='drbl_stage1_180x320/drbl_stage1_180x320.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )
```

## DRBL Stage1 with TensorFlow Lite in Python
```
python demo_DRBL_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='drbl_stage1_180x320/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )
```


