#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from argparse import ArgumentParser

"""
nms_decode_boxes_y1x1y2x2score [1, boxes, 5]

[0] -> y1
[1] -> x1
[2] -> y2
[3] -> x2
[4] -> score
"""


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        boxes = x[..., :4] # y1x1y2x2 [n, boxes, 4]
        scores = x[..., 4:5] # [n, boxes, 1]
        scores = scores.permute(0,2,1)# [n, 1, boxes]
        return boxes, scores


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-o',
        '--opset',
        type=int,
        default=12,
        help='onnx opset'
    )
    parser.add_argument(
        '-b',
        '--batches',
        type=str,
        default='N',
        help='batch size'
    )
    parser.add_argument(
        '-x',
        '--boxes',
        type=str,
        default='boxes',
        help='boxes'
    )
    args = parser.parse_args()

    model = Model()

    MODEL = f'boxes_scores'
    OPSET=args.opset
    BATCHES = args.batches
    BOXES = args.boxes

    onnx_file = f"{MODEL}_{BOXES}.onnx"
    x = torch.randn(1, 10, 5)

    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names = ['nms_decode_boxes_y1x1y2x2score'],
        output_names=['nms_y1x1y2x2','nms_scores'],
        dynamic_axes={
            'nms_decode_boxes_y1x1y2x2score' : {0: BATCHES, 1: 'boxes'},
            'nms_y1x1y2x2' : {0: BATCHES, 1: 'boxes', 2: '4'},
            'nms_scores' : {0: BATCHES, 1: '1', 2: 'boxes'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

