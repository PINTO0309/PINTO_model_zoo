# Demo projects

## HTNet with ONNX Runtime in Python
```
python demo_HTNet_onnx.py
```

If you want to change the model, specify it with an argument.
```python
parser.add_argument(
    '-mod',
    '--model',
    type=str,
    default='htnet_1x17x2_without_norm.onnx',
)
parser.add_argument(
    '-p',
    '--provider',
    type=str,
    default='cpu',
    choices=['cpu','cuda','tensorrt'],
)
```
