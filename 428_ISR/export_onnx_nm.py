import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SwinTransformer(nn.Module):

    def __init__(self, num_features=512):
        super(SwinTransformer, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224')
        self.num_features = num_features
        self.feat = nn.Linear(1024, num_features) if num_features > 0 else None

    def forward(self, x, y):
        x1 = self.model.forward_features(x)
        if not self.feat is None:
            x1 = self.feat(x1)
        x2 = self.model.forward_features(y)
        if not self.feat is None:
            x2 = self.feat(x2)
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        z = torch.matmul(x1, x2.T)
        return z

if __name__ == '__main__':
    model = SwinTransformer(num_features=512).cuda()
    weight_path = 'swin_base_patch4_window7_224.pth'
    weight = torch.load(weight_path)
    model.load_state_dict(weight['state_dict'], strict=True)
    model.eval()
    model.cpu()

    import onnx
    from onnxsim import simplify
    RESOLUTION = [
        [224,224],
    ]
    MODEL = f'isr'
    OPSET = 11
    for H, W in RESOLUTION:
        onnx_file = f"{MODEL}_NMx3x{H}x{W}_{OPSET}.onnx"
        x = torch.randn(2, 3, H, W).cpu()
        y = torch.randn(3, 3, H, W).cpu()
        torch.onnx.export(
            model,
            args=(x, y),
            f=onnx_file,
            opset_version=OPSET,
            input_names=['input_base', 'input_target'],
            output_names=['output'],
            dynamic_axes={
                'input_base' : {0: 'N'},
                'input_target' : {0: 'M'},
                'output' : {0: 'N', 1: 'M'},
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
