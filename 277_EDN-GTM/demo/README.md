# Demo projects

## EDN-GTM with ONNX Runtime in Python
```
python demo_EDN-GTM_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
      "--device",
      type=int,
      default=0
    )
    parser.add_argument(
      "--movie",
      type=str,
      default=None
    )
    parser.add_argument(
        "--model",
        type=str,
        default='ihaze_generator_384x640/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='384,640',
    )
```

- ONNX 384x640 - Input movie 360x640 - ONNX TensorRT Execution Provider Float16

