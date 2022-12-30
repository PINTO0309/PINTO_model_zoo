# Demo projects

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

- CPU, CUDA, TensorRT

    https://user-images.githubusercontent.com/33194443/210041220-b1810699-f04d-4c94-9eee-9d58754fbe4f.mp4

## RFB320 + Dense-HeadPose Estimation with ONNX Runtime in Python
```
python demo_rfb320_denseface_sparse_pose_with_postprocess_onnx.py
```
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
    '-modrfb',
    '--model_rfb',
    type=str,
    default='RFB-320_240x320_post.onnx',
)
parser.add_argument(
    '-modden',
    '--model_dense',
    type=str,
    default='dense_face_Nx3x120x120.onnx',
)
parser.add_argument(
    '-modspa',
    '--model_sparse',
    type=str,
    default='sparse_face_Nx3x120x120.onnx',
)
parser.add_argument(
    '-m',
    '--mode',
    type=str,
    default='pose',
    choices=['pose','sparse','dense'],
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
- Pose - CPU, CUDA

    https://user-images.githubusercontent.com/33194443/210068071-29ed718f-bafc-4fac-97d2-4fcc83963ccd.mp4

- Sparse - CPU, CUDA

    https://user-images.githubusercontent.com/33194443/210068130-75f9a928-7188-4cd4-8606-3090afe65fd0.mp4

