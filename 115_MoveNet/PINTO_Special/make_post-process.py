#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify
from argparse import ArgumentParser

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        keypoints_with_scores,
        image_height,
        image_width,
    ):
        image_height = image_height.to(torch.float32)
        image_width = image_width.to(torch.float32)
        """
        keypoints_with_scores default size: [1, 6, 56]

        6 = number of people

        56 = [
            kp0_y, kp0_x, kp0_score,
                :
            kp16_y, kp16_x, kp16_score,
            bbox_y1, bbox_x1, bbox_y2, bbox_x2, bbox_score
        ]
        """
        keypoints_y_x_scores = keypoints_with_scores[..., 0:51]
        keypoints_y = keypoints_y_x_scores[..., 0::3] * image_height
        keypoints_x = keypoints_y_x_scores[..., 1::3] * image_width

        keypoints_y_x_scores[..., 0::3] = keypoints_x[..., 0::1]
        keypoints_y_x_scores[..., 1::3] = keypoints_y[..., 0::1]

        """
        keypoints_y = [1, 6, [y0, y0, ..., y16]]
        keypoints_x = [1, 6, [x0, x0, ..., x16]]
        keypoint_scores = [1, 6, [score0, score0, ..., score16]]
        """
        bboxes_y1 = keypoints_with_scores[..., 51:52] * image_height
        bboxes_x1 = keypoints_with_scores[..., 52:53] * image_width
        bboxes_y2 = keypoints_with_scores[..., 53:54] * image_height
        bboxes_x2 = keypoints_with_scores[..., 54:55] * image_width
        bboxes_score = keypoints_with_scores[..., 55:56]

        keypoints_bboxes = torch.cat(
            [
                keypoints_y_x_scores,
                bboxes_x1,
                bboxes_y1,
                bboxes_x2,
                bboxes_y2,
                bboxes_score,
            ],
            dim=2,
        )
        return keypoints_bboxes


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
        '-n',
        '--num_person',
        type=int,
        default=6,
        help='number of person'
    )
    parser.add_argument(
        '-oh',
        '--original_image_height',
        type=int,
        default=480,
        help='Height of the cameras input image (before rescaling)'
    )
    parser.add_argument(
        '-ow',
        '--original_image_width',
        type=int,
        default=640,
        help='Width of the cameras input image (before rescaling)'
    )
    args = parser.parse_args()

    model = Model()

    MODEL = f'post_process'
    OPSET=args.opset
    BATCHES = args.batches
    NUMPERSON = args.num_person
    ORIGINAL_IMAGE_HEIGHT = args.original_image_height
    ORIGINAL_IMAGE_WIDTH = args.original_image_width

    """
    56 = [
        kp0_y, kp0_x, kp0_score,
            :
        kp16_y, kp16_x, kp16_score,
        bbox_y1, bbox_x1, bbox_y2, bbox_x2, bbox_score
    ]
    """
    # default: [1, 6, 56]
    keypoints_with_scores = torch.arange(
        BATCHES * NUMPERSON * 56,
        dtype=torch.float32
    ).view(BATCHES, NUMPERSON, 56)
    image_height = torch.tensor(ORIGINAL_IMAGE_HEIGHT, dtype=torch.int64)
    image_width = torch.tensor(ORIGINAL_IMAGE_WIDTH, dtype=torch.int64)

    onnx_file = f"{MODEL}_p{NUMPERSON}.onnx"
    torch.onnx.export(
        model,
        args=(
            keypoints_with_scores,
            image_height,
            image_width,
        ),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            'pp_input',
            'original_image_height',
            'original_image_width',
        ],
        output_names=[
            'batch_persons_kpxkpykpscore_x17_bx1by1bx2by2bscore',
        ],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)


    onnx_file = f"{MODEL}_p{NUMPERSON}_N.onnx"
    torch.onnx.export(
        model,
        args=(
            keypoints_with_scores,
            image_height,
            image_width,
        ),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            'pp_input',
            'original_image_height',
            'original_image_width',
        ],
        output_names=[
            'batch_persons_kpxkpykpscore_x17_bx1by1bx2by2bscore',
        ],
        dynamic_axes={
            'pp_input' : {0: 'batch'},
            'batch_persons_kpxkpykpscore_x17_bx1by1bx2by2bscore': {0: 'batch'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)