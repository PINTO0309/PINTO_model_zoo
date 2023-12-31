import torch
import onnx
from onnxsim import simplify
from fast_reid.fast_reid_interfece import FastReIDInterface

MODELS = [
    ['mot17_sbs_S50', 'fast_reid/configs/MOT17/sbs_S50.yml'],
    ['mot20_sbs_S50', 'fast_reid/configs/MOT20/sbs_S50.yml'],
]


for model, config in MODELS:
    weight = f'pretrained/{model}.pth'
    reid_model = FastReIDInterface(config, weight, 'cpu')
    reid_model.model.eval()
    reid_model.model.cpu()

    RESOLUTION = [
        [384,128],
        [352,128],
        [320,128],
        [288,128],
        [256,128],
    ]

    for H, W in RESOLUTION:
        onnx_file = f"{model}_1x3x{H}x{W}.onnx"
        x = torch.randn(1, 3, H, W).cpu()
        torch.onnx.export(
            reid_model.model,
            args=(x),
            f=onnx_file,
            opset_version=11,
            input_names=['image'],
            output_names=['feature'],
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

        onnx_file = f"{model}_Nx3x{H}x{W}.onnx"
        x = torch.randn(1, 3, H, W).cpu()
        torch.onnx.export(
            reid_model.model,
            args=(x),
            f=onnx_file,
            opset_version=11,
            input_names=['images'],
            output_names=['features'],
            dynamic_axes={
                'images' : {0: 'N'},
                'features' : {0: 'N'},
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
