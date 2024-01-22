#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from argparse import ArgumentParser

class Model(nn.Module):
    def __init__(self, strides: int):
        super(Model, self).__init__()
        self.strides = strides

    def forward(self, part_heatmaps_input: torch.Tensor, mask_for_colored_input: torch.Tensor):
        part_heatmaps_input = part_heatmaps_input.permute(0,3,1,2)
        resized_h = self.strides * part_heatmaps_input.shape[2]
        resized_w = self.strides * part_heatmaps_input.shape[3]

        resized_part_heatmaps = \
            torch.nn.functional.interpolate(
                part_heatmaps_input,
                size=(resized_h, resized_w),
                mode='bilinear',
                align_corners=True,
            )
        sigmoid_part_heatmaps = torch.sigmoid(resized_part_heatmaps)
        argmax_part_heatmaps = torch.argmax(sigmoid_part_heatmaps, dim=1, keepdim=True)
        zero_mask_values = torch.zeros_like(argmax_part_heatmaps)
        colored_mask = \
            torch.where(
                mask_for_colored_input.to(torch.bool),
                argmax_part_heatmaps,
                zero_mask_values
            )
        return colored_mask

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
        '-sh',
        '--seg_height',
        type=int,
        default=30,
        help='input height'
    )
    parser.add_argument(
        '-sw',
        '--seg_width',
        type=int,
        default=40,
        help='input width'
    )
    parser.add_argument(
        '-s',
        '--strides',
        type=int,
        default=16,
        help='strides'
    )
    args = parser.parse_args()

    MODEL = f'02_colored_segment_mask'
    OPSET: int = args.opset
    BATCHES: int = args.batches
    H: int = args.seg_height
    W: int = args.seg_width
    STRIDES: int = args.strides

    model = Model(
        strides=STRIDES,
    )

    onnx_file = f"{MODEL}_{BATCHES}x3x{H*STRIDES}x{W*STRIDES}.onnx"
    part_heatmaps_input = torch.randn(BATCHES, H, W, 24)
    mask_for_colored_input = torch.randn(BATCHES, 1, H*STRIDES, W*STRIDES)

    torch.onnx.export(
        model,
        args=(part_heatmaps_input, mask_for_colored_input),
        f=onnx_file,
        opset_version=OPSET,
        input_names = ['part_heatmaps_input', 'mask_for_colored_input'],
        output_names=['colored_mask_classid'],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
