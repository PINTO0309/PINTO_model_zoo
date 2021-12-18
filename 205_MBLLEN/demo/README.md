# Demo projects

## MBLLEN with ONNX Runtime in Python
```
python demo_MBLLEN_onnx.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.
```python
    parser.add_argument(
        "--model",
        type=str,
        default=
        'saved_model_Syn_img_lowlight_withnoise_180x320/model_float32.onnx',
    )

    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )
```

## MBLLEN with TensorFlow Lite in Python
```
python demo_MBLLEN_tflite.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.
```python
    parser.add_argument(
        "--model",
        type=str,
        default=
        'saved_model_Syn_img_lowlight_withnoise_180x320/model_float16_quant.tflite',
    )

    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )
```


