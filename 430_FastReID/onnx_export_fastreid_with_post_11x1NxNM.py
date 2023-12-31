import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnxsim import simplify
from fast_reid.fast_reid_interfece import FastReIDInterface

MODELS = [
    ['mot17_sbs_S50', 'fast_reid/configs/MOT17/sbs_S50.yml'],
    ['mot20_sbs_S50', 'fast_reid/configs/MOT20/sbs_S50.yml'],
]


class FastReIDInterfaceWithPost(nn.Module):
    def __init__(
        self,
        config,
        weight,
    ):
        super(FastReIDInterfaceWithPost, self).__init__()
        frid_model_a = FastReIDInterface(config, weight, 'cpu')
        frid_model_b = FastReIDInterface(config, weight, 'cpu')
        self.frid_model_a = frid_model_a.model
        self.frid_model_b = frid_model_b.model

    def forward(self, x, y):
        x = self.frid_model_a(x)
        x[torch.isinf(x)] = 1.0
        x_features = F.normalize(x, dim=1)
        y = self.frid_model_b(y)
        y[torch.isinf(y)] = 1.0
        y_features = F.normalize(y, dim=1)
        similarity = x_features.matmul(y_features.transpose(1, 0))
        return similarity


for model, config in MODELS:
    weight = f'pretrained/{model}.pth'
    reid_model = FastReIDInterfaceWithPost(config, weight)
    reid_model.eval()
    reid_model.cpu()

    RESOLUTION = [
        [384,128],
        [352,128],
        [320,128],
        [288,128],
        [256,128],
    ]

    for H, W in RESOLUTION:
        onnx_file = f"{model}_11x3x{H}x{W}_post.onnx"
        x = torch.randn(1, 3, H, W).cpu()
        y = torch.randn(1, 3, H, W).cpu()
        torch.onnx.export(
            reid_model,
            args=(x,y),
            f=onnx_file,
            opset_version=11,
            input_names=['base_image', 'target_image'],
            output_names=['similarity'],
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

        onnx_file = f"{model}_1Nx3x{H}x{W}_post.onnx"
        x = torch.randn(1, 3, H, W).cpu()
        y = torch.randn(1, 3, H, W).cpu()
        torch.onnx.export(
            reid_model,
            args=(x,y),
            f=onnx_file,
            opset_version=11,
            input_names=['base_image', 'target_images'],
            output_names=['similarities'],
            dynamic_axes={
                'target_images' : {0: 'N'},
                'similarities' : {1: 'N'},
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

        onnx_file = f"{model}_NMx3x{H}x{W}_post.onnx"
        x = torch.randn(1, 3, H, W).cpu()
        y = torch.randn(1, 3, H, W).cpu()
        torch.onnx.export(
            reid_model,
            args=(x,y),
            f=onnx_file,
            opset_version=11,
            input_names=['base_images', 'target_images'],
            output_names=['similarities'],
            dynamic_axes={
                'base_images' : {0: 'N'},
                'target_images' : {0: 'M'},
                'similarities' : {0: 'N', 1: 'M'},
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

