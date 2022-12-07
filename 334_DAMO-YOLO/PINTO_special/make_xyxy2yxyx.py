#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from argparse import ArgumentParser

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, boxes, scores):
        """
        boxes:  [N, BOXES, 4]
        scores: [N, BOXES, CLASSES]
        """
        boxes_xyxy = boxes
        boxes_yxyx = torch.cat(
            [boxes[..., 1:2],boxes[..., 0:1],boxes[..., 3:4],boxes[..., 2:3]],
            dim=2,
        )
        scores_ncb = scores.permute(0,2,1)
        return boxes_xyxy, boxes_yxyx, scores_ncb


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
        default=756,
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

    MODEL = f'boxes_scores'
    OPSET=args.opset
    BATCHES = args.batches
    BOXES = args.boxes
    CLASSES = args.classes

    onnx_file = f"{MODEL}_{BOXES}.onnx"
    boxes = torch.randn(BATCHES, BOXES, 4)
    scores = torch.randn(BATCHES, BOXES, CLASSES)

    torch.onnx.export(
        model,
        args=(boxes,scores),
        f=onnx_file,
        opset_version=OPSET,
        input_names = [
            'post_boxes',
            'post_scores',
        ],
        output_names=[
            'boxes_xyxy_for_final_boxes',
            'boxes_yxyx_for_nms',
            'scores_for_nms'
        ],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)