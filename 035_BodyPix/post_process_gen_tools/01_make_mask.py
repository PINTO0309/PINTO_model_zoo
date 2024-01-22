#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from argparse import ArgumentParser

class Model(nn.Module):
    def __init__(self, strides: int, threshold: float):
        super(Model, self).__init__()
        self.strides = strides
        self.threshold = threshold

    def forward(self, segments: torch.Tensor):
        segments = segments.permute(0,3,1,2)
        resized_h = self.strides * segments.shape[2]
        resized_w = self.strides * segments.shape[3]

        if resized_h <= resized_w:
            resized_segments = \
                torch.nn.functional.interpolate(
                    segments,
                    size=(resized_h, resized_w),
                    mode='bilinear',
                    align_corners=True,
                )
        else:
            resized_segments = \
                torch.nn.functional.interpolate(
                    segments,
                    scale_factor=self.strides,
                    mode='bilinear',
                    align_corners=True,
                )
            self.threshold = 0.20
        sigmoid_segments = torch.sigmoid(resized_segments)
        mask_one_channel = torch.where(sigmoid_segments < self.threshold, torch.tensor(0.0, dtype=torch.float32), torch.tensor(255.0, dtype=torch.float32))

        mask_full_channel = torch.cat([mask_one_channel, mask_one_channel, mask_one_channel], dim=1)
        mask_for_colored = mask_one_channel / 255.0
        return mask_full_channel, mask_for_colored

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
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.60,
        help='threshold'
    )
    args = parser.parse_args()

    MODEL = f'01_segment_mask'
    OPSET: int = args.opset
    BATCHES: int = args.batches
    H: int = args.seg_height
    W: int = args.seg_width
    STRIDES: int = args.strides
    THRESHOLD: float = args.threshold

    model = Model(
        strides=STRIDES,
        threshold=THRESHOLD,
    )

    onnx_file = f"{MODEL}_{BATCHES}x3x{H*STRIDES}x{W*STRIDES}.onnx"
    segments = torch.randn(BATCHES, H, W, 1)

    torch.onnx.export(
        model,
        args=(segments),
        f=onnx_file,
        opset_version=OPSET,
        input_names = ['mask_input'],
        output_names=['foreground_mask_zero_or_255', 'mask_for_colored_output'],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
