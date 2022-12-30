# [WIP] Demo projects

## RFB320 with ONNX Runtime in Python
```
python demo_rfb320_with_postprocess_onnx.py
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
    default='RFB-320_240x320_post.onnx',
)
parser.add_argument(
    '-p',
    '--provider',
    type=str,
    default='cpu',
    choices=['cpu','cuda','tensorrt'],
)
parser.add_argument(
    '-s',
    '--score_th',
    type=float,
    default=0.7,
)
```

https://user-images.githubusercontent.com/33194443/210041220-b1810699-f04d-4c94-9eee-9d58754fbe4f.mp4
