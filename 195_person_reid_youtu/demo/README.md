# Demo projects

## person_reid_youtu with ONNX Runtime in Python
```
python demo_person_reid_youtu_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_person_reid_youtu/person_reid_youtu_2021nov.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,128',
    )
```

## person_reid_youtu with TensorFlow Lite in Python
```
python demo_person_reid_youtu_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_person_reid_youtu/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,128',
    )
```
## person_reid_youtu with ONNX and Tensorflowjs in Javascript(Typescript)
https://w-okada.github.io/yolox-onnx-test/

[movie](https://twitter.com/DannadoriYellow/status/1597369995302178817)
