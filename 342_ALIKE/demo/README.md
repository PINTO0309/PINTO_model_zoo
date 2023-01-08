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
parser.add_argument(
    '-s',
    '--skip_frame_count',
    type=int,
    default=0,
    help='skip_frame_count+1 value of whether the feature point is compared to the previous frame.'
)
parser.add_argument(
    '-oeed',
    '--output_exclude_euclidean_distance',
    type=float,
    default=150,
    help='Output Exclude Euclidean distance.'
)
```

- TensorRT - skip_frame_count=0

    https://user-images.githubusercontent.com/33194443/211139176-2b303feb-8ac5-4cb6-af55-30a336e144ff.mp4

- TensorRT - skip_frame_count=1

    https://user-images.githubusercontent.com/33194443/211182741-998923b6-857c-435a-b173-bbeb27aae457.mp4

- TensorRT - skip_frame_count=2

    https://user-images.githubusercontent.com/33194443/211182696-0b4cbd42-59dd-4c70-aa61-1a3b98cf1166.mp4


