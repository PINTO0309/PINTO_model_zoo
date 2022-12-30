import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self._anchors_wh = torch.from_numpy(np.load('anchors_wh.npy')).to(torch.float32).clone()
        self._anchors_xy = torch.from_numpy(np.load('anchors_xy.npy')).to(torch.float32).clone()

    def forward(self,boxes,scores):
        center_xy = boxes[..., :2] * torch.tensor(0.1, dtype=torch.float32) * self._anchors_wh + self._anchors_xy
        center_wh = torch.exp(boxes[..., 2:] * torch.tensor(0.2, dtype=torch.float32)) * self._anchors_wh / 2
        start_xy = center_xy - center_wh
        end_xy = center_xy + center_wh
        boxes_cat = torch.cat([start_xy, end_xy], dim=-1)
        decoded_boxes = torch.clip(boxes_cat, 0.0, 1.0)
        decoded_boxes_y1x1y2x2 = torch.cat(
            [
                decoded_boxes[..., 1:2],
                decoded_boxes[..., 0:1],
                decoded_boxes[..., 3:4],
                decoded_boxes[..., 2:3],
            ],
            dim=2
        )
        return decoded_boxes, decoded_boxes_y1x1y2x2, scores[..., 1:2].permute(0,2,1)

model = Model()
model.eval()
model.cpu()
x = torch.randn(1,4420,4).cpu()
y = torch.randn(1,4420,2).cpu()

onnx_file = f'decode_boxes.onnx'
torch.onnx.export(
    model,
    args=(x,y),
    f=onnx_file,
    opset_version=11,
    input_names=['decode_boxes_input','decode_scores_input'],
    output_names=['decode_boxes_x1y1x2y2','decode_boxes_y1x1y2x2','decode_scores'],
)
import onnx
from onnxsim import simplify
model_onnx2 = onnx.load(onnx_file)
model_simp, check = simplify(model_onnx2)
onnx.save(model_simp, onnx_file)
