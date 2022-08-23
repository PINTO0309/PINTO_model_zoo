import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        output = torch.round(x)
        return output

if __name__ == "__main__":
    model = Model()

    import onnx
    from onnxsim import simplify
    MODEL = f'round_left_right'
    onnx_file = f"{MODEL}_1x3x224x224.onnx"
    x = torch.randn(1, 1)
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=11,
        input_names = ['round_input'],
        output_names=['lefthand_0_or_righthand_1'],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)



    onnx_file = f"{MODEL}_Nx3x224x224.onnx"
    x = torch.randn(1, 1)
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=11,
        input_names=['round_input'],
        output_names=['lefthand_0_or_righthand_1'],
        dynamic_axes={
            'round_input' : {0: 'N'},
            'lefthand_0_or_righthand_1' : {0: 'N'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)