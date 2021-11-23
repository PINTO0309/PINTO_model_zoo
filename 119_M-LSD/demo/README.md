# Demo projects

## M-LSD (Lines/Squares) with ONNX Runtime in Python
```
python demo_mlsd-lines_onnx.py
```
```
python demo_mlsd-squares_onnx.py
```


If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.
```python
    parser.add_argument("--model",
                        type=str,
                        default='saved_model_320x320_tiny/model_float32.onnx')
    parser.add_argument("--input_shape", type=str, default='320,320')
```

## M-LSD (Lines/Squares) with TensorFlow Lite in Python
```
python ddemo_mlsd-lines_tflite.py
```
```
python demo_mlsd-squares_tflite.py
```

If you want to change the model, specify it with an argument.<br>
Check the source code for other argument specifications.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_320x320_tiny/model_float16_quant.tflite')
    parser.add_argument("--input_shape", type=str, default='320,320')
```


