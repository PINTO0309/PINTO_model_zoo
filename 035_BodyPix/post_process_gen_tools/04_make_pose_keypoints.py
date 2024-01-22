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

    def forward(self, heatmaps_input: torch.Tensor, offsets_input: torch.Tensor):
        heatmaps_input = heatmaps_input.permute(0,3,1,2)
        offsets_input = offsets_input.permute(0,3,1,2)

        base_points_y, base_points_x = \
            torch.meshgrid(
                torch.arange(0, heatmaps_input.shape[2] * self.strides, self.strides, dtype=torch.float32),
                torch.arange(0, heatmaps_input.shape[3] * self.strides, self.strides, dtype=torch.float32)
            )
        base_points_y = base_points_y.reshape(heatmaps_input.shape[0], 1, heatmaps_input.shape[2], heatmaps_input.shape[3]).repeat(heatmaps_input.shape[0], 17, 1, 1)
        base_points_x = base_points_x.reshape(heatmaps_input.shape[0], 1, heatmaps_input.shape[2], heatmaps_input.shape[3]).repeat(heatmaps_input.shape[0], 17, 1, 1)
        y_offsets = offsets_input[:, 0:17, ...]
        x_offsets = offsets_input[:, 17:34, ...]
        base_points_y += y_offsets
        base_points_x += x_offsets

        base_points_x = base_points_x.reshape(heatmaps_input.shape[0], 17, heatmaps_input.shape[2], heatmaps_input.shape[3], 1)
        base_points_y = base_points_y.reshape(heatmaps_input.shape[0], 17, heatmaps_input.shape[2], heatmaps_input.shape[3], 1)
        bkhw_xy = torch.cat([base_points_x, base_points_y], dim=4)
        bkhw_xy = bkhw_xy.to(torch.int64)

        argmax_heatmaps_input = heatmaps_input.argmax(dim=1, keepdim=True)
        n,c,h,w = argmax_heatmaps_input.shape

        argmaxed_classids = argmax_heatmaps_input.reshape(n, c*h*w, 1).to(torch.float32)
        expanded_indices = argmax_heatmaps_input.reshape(n,c,h,w,1).expand(-1, -1, -1, -1, 2)
        extracted_tensor = torch.gather(bkhw_xy, 1, expanded_indices)
        n,c,h,w,xy = extracted_tensor.shape
        reshaped_tensor = extracted_tensor.reshape(n, c*h*w, xy).float()
        n,c,h,w = argmax_heatmaps_input.shape
        scores = torch.sigmoid(torch.gather(heatmaps_input, 1, argmax_heatmaps_input).reshape(n, c*h*w, 1))
        combined_tensor = torch.cat((argmaxed_classids, scores, reshaped_tensor), dim=2)

        return combined_tensor

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

    MODEL = f'04_pose_keypoints'
    OPSET: int = args.opset
    BATCHES: int = args.batches
    H: int = args.seg_height
    W: int = args.seg_width
    STRIDES: int = args.strides

    model = Model(
        strides=STRIDES,
    )

    onnx_file = f"{MODEL}_{BATCHES}x3x{H*STRIDES}x{W*STRIDES}.onnx"
    heatmaps = torch.randn(BATCHES, H, W, 17)
    offsets = torch.randn(BATCHES, H, W, 34)

    torch.onnx.export(
        model,
        args=(heatmaps, offsets),
        f=onnx_file,
        opset_version=OPSET,
        input_names = ['heatmaps_input', 'offsets_input'],
        output_names=['keypoints_classidscorexy'],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
