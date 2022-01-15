# Demo projects

## Single-Image-Desnowing-HDCWNet with ONNX Runtime in Python
```
python demo_Single-Image-Desnowing-HDCWNet_onnx.py
```

If you want to change the model, specify it with an argument.<br>
Note: The ONNX file will fail to load.<Br>
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_batch_1/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='512,672',
    )
```

## Single-Image-Desnowing-HDCWNet with TensorFlow Lite in Python
```
python demo_Single-Image-Desnowing-HDCWNet_tflite.py
```

If you want to change the model, specify it with an argument.<br>
Note: You need TensorFlow-Lite with FlexDelegate enabled.<Br>
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_batch_1/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='512,672',
    )
```

## About sample images
The sample image uses the image of "[PAKUTASO](https://www.pakutaso.com/)".<br>
If you want to use the image itself for another purpose, you must follow the [userpolicy of PAKUTASO](https://www.pakutaso.com/userpolicy.html).
