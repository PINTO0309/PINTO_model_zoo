# Demo projects

## ALIKE with ONNX Runtime in Python
```
python demo_alike_with_postprocess_onnx.py
```

If you want to change the model, specify it with an argument.
```python
parser.add_argument(
    '-d',
    '--device',
    type=int,
    default=0,
)
parser.add_argument(
    '-mov',
    '--movie',
    type=str,
    default=None,
)
parser.add_argument(
    '-mod',
    '--model',
    type=str,
    default='alike_t_opset16_480x640_post.onnx',
)
parser.add_argument(
    '-p',
    '--provider',
    type=str,
    default='cpu',
    choices=['cpu','cuda','tensorrt'],
)
```

- TensorRT

    https://user-images.githubusercontent.com/33194443/211139176-2b303feb-8ac5-4cb6-af55-30a336e144ff.mp4
