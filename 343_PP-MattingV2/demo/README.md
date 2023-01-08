# Demo projects

## PP-MattingV2 with ONNX Runtime in Python
```
python demo_ppmattingv2_onnx.py
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
    default='ppmattingv2_stdc1_human_480x640.onnx',
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
    '--score_threshold',
    type=float,
    default=0.65,
)
```

- TensorRT


