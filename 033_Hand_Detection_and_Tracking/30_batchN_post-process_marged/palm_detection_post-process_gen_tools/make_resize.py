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
        input_names=['resize1_input'],
        output_names=['resize1_output'],
        dynamic_axes={
            'resize1_input' : {0: 'batch'},
            'resize1_output' : {0: 'batch'},
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
        input_names=['resize2_input'],
        output_names=['resize2_output'],
        dynamic_axes={
            'resize2_input' : {0: 'batch'},
            'resize2_output' : {0: 'batch'},
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
        input_names=['resize3_input'],
        output_names=['resize3_output'],
        dynamic_axes={
            'resize3_input' : {0: 'batch'},
            'resize3_output' : {0: 'batch'},
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
        input_names=['resize4_input'],
        output_names=['resize4_output'],
        dynamic_axes={
            'resize4_input' : {0: 'batch'},
            'resize4_output' : {0: 'batch'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
