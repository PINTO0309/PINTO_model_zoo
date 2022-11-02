#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from argparse import ArgumentParser


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, batch, classid, y1x1y2x2):
        batchno_classid_y1x1y2x2_cat = torch.cat([batch, classid, y1x1y2x2], dim=1)
        return batchno_classid_y1x1y2x2_cat


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

    MODEL = f'nms_batchno_classid_y1x1y2x2_cat'

    onnx_file = f"{MODEL}.onnx"
    OPSET=args.opset

    x1 = torch.ones([1, 1], dtype=torch.int64)
    x2 = torch.ones([1, 1], dtype=torch.int64)
    x3 = torch.ones([1, 4], dtype=torch.int64)

    torch.onnx.export(
        model,
        args=(x1,x2,x3),
        f=onnx_file,
        opset_version=OPSET,
        input_names=['cat_batch','cat_classid','cat_y1x1y2x2'],
        output_names=['batchno_classid_y1x1y2x2'],
        dynamic_axes={
            'cat_batch': {0: 'N'},
            'cat_classid': {0: 'N'},
            'cat_y1x1y2x2': {0: 'N'},
            'batchno_classid_y1x1y2x2': {0: 'N'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

