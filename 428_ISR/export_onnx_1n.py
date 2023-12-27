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

    def forward(self, x):
        x1 = x[0:1, ...]
        x2 = x[1:, ...]

        x1 = self.model.forward_features(x1)
        if not self.feat is None:
            x1 = self.feat(x1)
        x2 = self.model.forward_features(x2)
        if not self.feat is None:
            x2 = self.feat(x2)
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        y = torch.matmul(x1, x2.T)
        return y

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
        onnx_file = f"{MODEL}_Nx3x{H}x{W}_{OPSET}.onnx"
        x = torch.randn(3, 3, H, W).cpu()
        torch.onnx.export(
            model,
            args=(x),
            f=onnx_file,
            opset_version=OPSET,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input' : {0: 'N'},
                'output' : {0: '1', 1: 'N'},
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

"""
************************************************** opset=18
sit4onnx -if isr_2x3x224x224_18.onnx -oep cpu

INFO: file: isr_2x3x224x224.onnx
INFO: providers: ['CPUExecutionProvider']
INFO: input_name.1: input shape: [2, 3, 224, 224] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  1153.2487869262695 ms
INFO: avg elapsed time per pred:  115.32487869262695 ms
INFO: output_name.1: output shape: [1, 1] dtype: float32

sit4onnx -if isr_2x3x224x224_18.onnx -oep cuda

INFO: file: isr_2x3x224x224_18.onnx
INFO: providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: input shape: [2, 3, 224, 224] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  168.84374618530273 ms
INFO: avg elapsed time per pred:  16.884374618530273 ms
INFO: output_name.1: output shape: [1, 1] dtype: float32

************************************************** opset=11
sit4onnx -if isr_2x3x224x224_11.onnx -oep cpu

INFO: file: isr_2x3x224x224_11.onnx
INFO: providers: ['CPUExecutionProvider']
INFO: input_name.1: input shape: [2, 3, 224, 224] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  1564.5437240600586 ms
INFO: avg elapsed time per pred:  156.45437240600586 ms
INFO: output_name.1: output shape: [1, 1] dtype: float32

sit4onnx -if isr_2x3x224x224_11.onnx -oep cuda

INFO: file: isr_2x3x224x224_11.onnx
INFO: providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: input shape: [2, 3, 224, 224] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  168.98679733276367 ms
INFO: avg elapsed time per pred:  16.898679733276367 ms
INFO: output_name.1: output shape: [1, 1] dtype: float32

sit4onnx -if isr_2x3x224x224_11.onnx -oep tensorrt

INFO: file: isr_2x3x224x224_11.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: input shape: [2, 3, 224, 224] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  49.81803894042969 ms
INFO: avg elapsed time per pred:  4.981803894042969 ms
INFO: output_name.1: output shape: [1, 1] dtype: float32
"""