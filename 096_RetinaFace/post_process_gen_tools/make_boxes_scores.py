#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from argparse import ArgumentParser

"""
prediction [1, 5040, 85]

80 classes

85

[0] -> center_x
[1] -> center_y
[2] -> width
[3] -> height
[4] -> box_score
[5]-[84] -> class_score
"""


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        boxes = x[..., :4] # xywh [n, boxes, 4]
        box_scores = x[..., 4:5] # [n, boxes, 1]
        class_scores = x[..., 5:] # [n, boxes, 80]
        scores = torch.sqrt(box_scores * class_scores)
        scores = scores.permute(0,2,1)
        return boxes, scores


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-o',
        '--opset',
        type=int,
        default=11,
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

    model = Model()

    MODEL = f'02_boxes_scores'
    OPSET=args.opset
    BATCHES = args.batches
    BOXES = args.boxes
    CLASSES = args.classes

    onnx_file = f"{MODEL}_{BOXES}.onnx"
    x = torch.randn(BATCHES, BOXES, CLASSES+5)

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

