import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        h = x.shape[2]
        w = x.shape[3]
        output = F.resize(img=x, size=(h*2, w*2))
        return output

if __name__ == "__main__":
    model = Model()

    import onnx
    MODEL = f'resize'

    H=4
    W=4
    onnx_file = f"{MODEL}_x2_Nx256x{H}x{W}.onnx"
    x = torch.randn(1, 256, H, W)
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input' : {0: 'batch'},
            'output' : {0: 'batch'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    H=6
    W=6
    onnx_file = f"{MODEL}_x2_Nx256x{H}x{W}.onnx"
    x = torch.randn(1, 256, H, W)
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input' : {0: 'batch'},
            'output' : {0: 'batch'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    H=8
    W=8
    onnx_file = f"{MODEL}_x2_Nx256x{H}x{W}.onnx"
    x = torch.randn(1, 256, H, W)
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input' : {0: 'batch'},
            'output' : {0: 'batch'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    H=12
    W=12
    onnx_file = f"{MODEL}_x2_Nx256x{H}x{W}.onnx"
    x = torch.randn(1, 256, H, W)
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input' : {0: 'batch'},
            'output' : {0: 'batch'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
