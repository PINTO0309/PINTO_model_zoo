# Note

https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/2f75d180-eea9-4da6-988f-ba76cd589f00

```
$ sit4onnx -if face_blendshapes.onnx -oep cpu
INFO: file: face_blendshapes.onnx
INFO: providers: ['CPUExecutionProvider']
INFO: input_name.1: input_points shape: [1, 146, 2] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  5.369663238525391 ms
INFO: avg elapsed time per pred:  0.5369663238525391 ms
INFO: output_name.1: output shape: [52] dtype: float32

$ sit4onnx -if face_blendshapes.onnx -oep cuda
INFO: file: face_blendshapes.onnx
INFO: providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: input_points shape: [1, 146, 2] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  28.800249099731445 ms
INFO: avg elapsed time per pred:  2.8800249099731445 ms
INFO: output_name.1: output shape: [52] dtype: float32

$ sit4onnx -if face_blendshapes.onnx -oep tensorrt
INFO: file: face_blendshapes.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: input_points shape: [1, 146, 2] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  3.676176071166992 ms
INFO: avg elapsed time per pred:  0.3676176071166992 ms
INFO: output_name.1: output shape: [52] dtype: float32
```
```python
import torch
import torch.nn as nn

class ReshapeLayer(nn.Module):
    def __init__(self):
        super(ReshapeLayer, self).__init__()

    def forward(self, x):
        x = torch.reshape(x, shape=[1, 52])
        return x

OPSET = 11
model = ReshapeLayer()
x = torch.randn(1, 52, 1, 1)
torch.onnx.export(
    model,
    x,
    f'reshape_{OPSET}.onnx',
    input_names=['input_reshape'],
    output_names=['output'],
    opset_version=OPSET,
)
```
```python
import torch
import torch.nn as nn
import numpy as np

class TileLayer(nn.Module):
    def __init__(self):
        super(TileLayer, self).__init__()

        x = np.load('tile_target_tensor.npy')
        self.x = torch.tensor(x)

    def forward(self, input_tensor):
        mul_ones = torch.ones((input_tensor.shape[0],1,1,1), dtype=self.x.dtype)
        return torch.cat([(self.x * mul_ones), input_tensor], dim=3)

OPSET = 11
model = TileLayer()
input_tensor = torch.randn([2,64,1,96], dtype=torch.float32)
torch.onnx.export(
    model,
    (input_tensor),
    f'tile_{OPSET}.onnx',
    input_names=['input_tile'],
    output_names=['output_tile'],
    opset_version=OPSET,
    dynamic_axes={
        'input_tile' : {0: 'N'},
        'output_tile' : {0: 'N'},
    }
)
```
