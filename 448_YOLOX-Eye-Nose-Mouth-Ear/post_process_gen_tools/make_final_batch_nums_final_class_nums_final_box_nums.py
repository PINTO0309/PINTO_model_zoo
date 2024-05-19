#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
import numpy as np
from onnxsim import simplify
from argparse import ArgumentParser

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        batch_nums = x[:, 0:1].to(torch.float32) # batch number
        class_nums = x[:, 1:2].to(torch.float32) # class ids
        box_nums = x[:, [0,2]] # batch number + box number
        return batch_nums, class_nums, box_nums


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

    MODEL = f'13_nms_final_batch_nums_final_class_nums_final_box_nums'
    OPSET=args.opset

    onnx_file = f"{MODEL}.onnx"
    x = torch.ones([1, 3], dtype=torch.int64)

    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names=['bc_input'],
        output_names=['final_batch_nums','final_class_nums','final_box_nums'],
        dynamic_axes={
            'bc_input': {0: 'N'},
            'final_batch_nums': {0: 'N'},
            'final_class_nums': {0: 'N'},
            'final_box_nums': {0: 'N'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

