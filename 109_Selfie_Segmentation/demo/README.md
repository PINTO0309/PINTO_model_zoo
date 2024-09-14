# Demo projects

## Selfie Segmenttion with ONNX Runtime in Python
```
python demo_selfie_segmentation_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_tflite_tfjs_tftrt_onnx_coreml/model_float32.onnx',
        help="Path to the ONNX model to be used.",
    )
```
