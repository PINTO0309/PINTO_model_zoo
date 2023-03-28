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

```
$ sit4onnx -if htnet_1x17x2_with_norm.onnx -oep cpu
INFO: file: htnet_1x17x2_with_norm.onnx
INFO: providers: ['CPUExecutionProvider']
INFO: input_name.1: batch_keypoints_absXY_1x17x2 shape: [1, 17, 2] dtype: float32
INFO: input_name.2: image_height shape: [] dtype: int64
INFO: input_name.3: image_width shape: [] dtype: int64
INFO: test_loop_count: 10
INFO: total elapsed time:  14.341115951538086 ms
INFO: avg elapsed time per pred:  1.4341115951538086 ms
INFO: output_name.1: output3d_1x17x3 shape: [1, 17, 3] dtype: float32

$ sit4onnx -if htnet_1x17x2_with_norm.onnx -oep cuda
INFO: file: htnet_1x17x2_with_norm.onnx
INFO: providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: batch_keypoints_absXY_1x17x2 shape: [1, 17, 2] dtype: float32
INFO: input_name.2: image_height shape: [] dtype: int64
INFO: input_name.3: image_width shape: [] dtype: int64
INFO: test_loop_count: 10
INFO: total elapsed time:  18.05400848388672 ms
INFO: avg elapsed time per pred:  1.8054008483886719 ms
INFO: output_name.1: output3d_1x17x3 shape: [1, 17, 3] dtype: float32

$ sit4onnx -if htnet_1x17x2_with_norm.onnx -oep tensorrt
INFO: file: htnet_1x17x2_with_norm.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: batch_keypoints_absXY_1x17x2 shape: [1, 17, 2] dtype: float32
INFO: input_name.2: image_height shape: [] dtype: int64
INFO: input_name.3: image_width shape: [] dtype: int64
INFO: test_loop_count: 10
INFO: total elapsed time:  5.74946403503418 ms
INFO: avg elapsed time per pred:  0.574946403503418 ms
INFO: output_name.1: output3d_1x17x3 shape: [1, 17, 3] dtype: float32
```

![lindan_pose](https://user-images.githubusercontent.com/33194443/228307811-757bbc12-2c45-4aa7-b877-97cc192376b6.png)

![messi_pose](https://user-images.githubusercontent.com/33194443/228307819-6e712c51-3a52-44b7-8075-f11c16dd9632.png)
