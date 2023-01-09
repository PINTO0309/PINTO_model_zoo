# Demo projects

## XYDeblur with ONNX Runtime in Python
```
python demo_xy_single_image_deblur_onnx.py
```

If you want to change the model, specify it with an argument.
```python
parser = argparse.ArgumentParser()
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
    default='xy_single_image_deblur_480x640.onnx',
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

    https://user-images.githubusercontent.com/33194443/211249484-15cc8f67-1691-4802-877c-d92087a7eec1.mp4


