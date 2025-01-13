#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from argparse import ArgumentParser

"""
prediction [1, 5040, 12]

[0] -> center_x
[1] -> center_y
[2] -> width
[3] -> height
[4]-[11] -> class_score
"""


class Model(nn.Module):
    def __init__(self, classes: int):
        super(Model, self).__init__()
        self.classes = classes

    def forward(self, x: torch.Tensor):
        boxes = x[..., 0:4, :] # xywh [n, 4, boxes]
        scores = x[..., 4:4+self.classes, :] # [n, 8, boxes]
        # scores = torch.sqrt(scores)
        boxes = boxes.permute(0,2,1)
        return boxes, scores


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-o',
        '--opset',
        type=int,
        default=13,
        help='onnx opset'
    )
    parser.add_argument(
        '-b',
        '--batches',
        type=int,
        default=1,
        help='batch size'
    )
    parser.add_argument(
        '-x',
        '--boxes',
        type=int,
        default=5040,
        help='boxes'
    )
    parser.add_argument(
        '-c',
        '--classes',
        type=int,
        default=80,
        help='classes'
    )
    args = parser.parse_args()

    MODEL = f'02_boxes_scores'
    OPSET=args.opset
    BATCHES = args.batches
    BOXES = args.boxes
    CLASSES = args.classes

    model = Model(classes=CLASSES)

    onnx_file = f"{MODEL}_{BOXES}.onnx"
    x = torch.randn(BATCHES, CLASSES+4, BOXES)

    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names = ['boxes_scores_input'],
        output_names=['boxes_cxcywh','scores'],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

