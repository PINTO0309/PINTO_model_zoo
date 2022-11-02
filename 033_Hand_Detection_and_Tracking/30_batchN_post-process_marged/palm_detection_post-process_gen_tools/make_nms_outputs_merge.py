#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from argparse import ArgumentParser


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, cat_score, cat_boxes):
        score_cx_cy_w_wristcenterxy_middlefingerxy_cat = torch.cat([cat_score, cat_boxes], dim=1)
        return score_cx_cy_w_wristcenterxy_middlefingerxy_cat


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-o',
        '--opset',
        type=int,
        default=11,
        help='onnx opset'
    )
    args = parser.parse_args()

    model = Model()

    MODEL = f'nms_scores_boxes_cat'

    onnx_file = f"{MODEL}.onnx"
    OPSET=args.opset

    x1 = torch.ones([1, 1], dtype=torch.float32)
    x2 = torch.ones([1, 7], dtype=torch.float32)

    torch.onnx.export(
        model,
        args=(x1,x2),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            'cat_score',
            'cat_boxes',
        ],
        output_names=[
            'score_cx_cy_w_wristcenterxy_middlefingerxy',
        ],
        dynamic_axes={
            'cat_score': {0: 'N'},
            'cat_boxes': {0: 'N'},
            'score_cx_cy_w_wristcenterxy_middlefingerxy': {0: 'N'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

