#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from argparse import ArgumentParser

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = x[:, [0,2]]
        return x


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
    args = parser.parse_args()

    model = Model()

    MODEL = f'gather'
    OPSET=args.opset
    BATCHES = args.batches

    onnx_file = f"gather.onnx"
    x = torch.ones([BATCHES, 3], dtype=torch.int64)

    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names = [
            'post_gather_input',
        ],
        output_names=[
            'post_gateher_output',
        ],
        dynamic_axes={
            'post_gather_input' : {0: 'N'},
            'post_gateher_output' : {0: 'N'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)