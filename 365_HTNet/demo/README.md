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
    default='htnet_1x17x2_with_norm.onnx',
)
parser.add_argument(
    '-p',
    '--provider',
    type=str,
    default='cpu',
    choices=['cpu','cuda','tensorrt'],
)
```

![lindan_pose](https://user-images.githubusercontent.com/33194443/228307811-757bbc12-2c45-4aa7-b877-97cc192376b6.png)

![messi_pose](https://user-images.githubusercontent.com/33194443/228307819-6e712c51-3a52-44b7-8075-f11c16dd9632.png)
