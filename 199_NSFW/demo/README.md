# Demo projects

## NSFW with ONNX Runtime in Python
```
python demo_nsfw_onnx.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.<br>
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_nsfw/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='224,224',
    )
```

## NSFW with TensorFlow Lite in Python
```
python demo_nsfw_tflite.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.<br>
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_nsfw/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='224,224',
    )
```

## About sample images
The sample image uses the image of "[PAKUTASO](https://www.pakutaso.com/)".<br>
If you want to use the image itself for another purpose, you must follow the [userpolicy of PAKUTASO](https://www.pakutaso.com/userpolicy.html).
