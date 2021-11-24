# Demo projects

## AnimeGANv2 with ONNX Runtime in Python
```
python demo_AnimeGANv2_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='../01_float32/saved_model_Paprika/model_float32_opt.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,256',
    )
```

## AnimeGANv2 with TensorFlow Lite in Python
```
python demo_AnimeGANv2_tflite.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default=
        '../05_float16_quantization/animeganv2_paprika_256x256_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,256',
    )
```


