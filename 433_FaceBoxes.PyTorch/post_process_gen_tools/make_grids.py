#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
import numpy as np
from onnxsim import simplify
from typing import List
from argparse import ArgumentParser

class Model(nn.Module):
    def __init__(self, img_size, strides):
        super(Model, self).__init__()

        self.img_size = img_size
        self.strides = strides

    def forward(self, x):
        hsizes = [self.img_size[0] // stride for stride in self.strides]
        wsizes = [self.img_size[1] // stride for stride in self.strides]

        grids = []
        expanded_strides = []

        for hsize, wsize, stride in zip(hsizes, wsizes, self.strides):
            ix = torch.arange(wsize)
            iy = torch.arange(hsize)
            yv, xv = torch.meshgrid([iy, ix], indexing='ij')
            grid = torch.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        x[..., :2] = (x[..., :2] + grids) * expanded_strides
        x[..., 2:4] = torch.exp(x[..., 2:4]) * expanded_strides
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
        '-ih',
        '--image_height',
        type=int,
        default=416,
        help='height'
    )
    parser.add_argument(
        '-iw',
        '--image_width',
        type=int,
        default=416,
        help='width'
    )
    parser.add_argument(
        '-s',
        '--strides',
        type=int,
        nargs='*',
        default=[8, 16, 32],
        help='strides'
    )
    parser.add_argument(
        '-x',
        '--boxes',
        type=int,
        default=3549,
        help='boxes'
    )
    parser.add_argument(
        '-c',
        '--classes',
        type=int,
        default=16,
        help='classes'
    )
    args = parser.parse_args()

    image_height: int = args.image_height
    image_width: int = args.image_width
    strides: List[int] = args.strides
    boxes: int = args.boxes
    classes: int = args.classes
    model = Model(img_size=[image_height, image_width], strides=strides)

    MODEL = f'01_grid'
    OPSET=args.opset

    onnx_file = f"{MODEL}_{boxes}.onnx"
    x = torch.randn(1, boxes, classes+5)

    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names = ['predictions'],
        output_names=['grid_output'],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)


