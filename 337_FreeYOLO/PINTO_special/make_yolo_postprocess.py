#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from typing import List
from argparse import ArgumentParser

from sor4onnx import rename

class Model(nn.Module):
    def __init__(self, input_shape, classes, strides):
        super(Model, self).__init__()

        self.input_shape: List[int] = input_shape
        self.classes: int = classes
        self.strides: List[int] = strides

    def _decode_boxes(
        self,
        anchors,
        pred_regs,
        expand_strides,
    ):
        # center of bbox
        pred_ctr_xy = anchors[..., 0:2] + pred_regs[..., 0:2] * expand_strides
        # size of bbox
        pred_box_wh = torch.exp(pred_regs[..., 2:4]) * expand_strides
        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        pred_box = torch.concatenate([pred_x1y1, pred_x2y2], axis=2)
        return pred_box


    def _generate_anchors(
        self,
        input_shape,
        strides,
    ):
        all_anchors = []
        all_expand_strides = []
        for stride in strides:
            # generate grid cells
            fmp_h, fmp_w = input_shape[2] // stride, input_shape[3] // stride
            anchor_x, anchor_y = torch.meshgrid(
                torch.arange(fmp_w),
                torch.arange(fmp_h),
            )
            # [H, W, 2]
            anchor_xy = torch.stack([anchor_x, anchor_y], axis=-1)
            shape = anchor_xy.shape[:2]
            # [H, W, 2] -> [HW, 2]
            anchor_xy = (anchor_xy.reshape(-1, 2) + 0.5) * stride
            all_anchors.append(anchor_xy)
            # expanded stride
            strides = torch.full((*shape, 1), stride)
            all_expand_strides.append(strides.reshape(-1, 1))

        anchors = torch.concatenate(all_anchors, axis=0)
        expand_strides = torch.concatenate(all_expand_strides, axis=0)
        return anchors, expand_strides


    def _postprocess(
        self,
        predictions,
    ):
        anchors, expand_strides = self._generate_anchors(
            self.input_shape,
            self.strides,
        )
        reg_preds = predictions[..., 0:4]
        obj_preds = predictions[..., 4:5]
        cls_preds = predictions[..., 5:5 + self.classes]
        scores = torch.sqrt(obj_preds * cls_preds)
        bboxes = self._decode_boxes(
            anchors=anchors,
            pred_regs=reg_preds,
            expand_strides=expand_strides,
        )
        return bboxes, scores


    def forward(self, x):
        bboxes, scores = self._postprocess(
            predictions=x,
        )
        return bboxes, scores


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-o',
        '--opset',
        type=int,
        default=11,
        help='onnx opset',
    )
    parser.add_argument(
        '-mis',
        '--model_input_shape',
        type=int,
        nargs=4,
        default=[1,3,192,320],
        help='model input shape N,C,H,W',
    )
    parser.add_argument(
        '-s',
        '--strides',
        type=int,
        nargs='+',
        default=[8,16,32],
        help='anchor strides',
    )
    parser.add_argument(
        '-c',
        '--classes',
        type=int,
        default=80,
        help='classes',
    )
    parser.add_argument(
        '-b',
        '--boxes',
        type=int,
        default=1260,
        help='boxes',
    )
    args = parser.parse_args()

    opset: int = args.opset
    model_input_shape: List[int] = args.model_input_shape
    strides: List[int] = args.strides
    classes: int = args.classes
    boxes: int = args.boxes

    model = Model(
        input_shape=model_input_shape,
        classes=classes,
        strides=strides,
    )

    x = torch.randn([1, boxes, 5+classes])
    onnx_file = f'postprocess_anchors_{boxes}.onnx'

    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=opset,
        input_names=['post_input'],
        output_names=['bboxes_xyxy', 'scores'],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    rename(
        old_new=['/Slice','post_Slice'],
        input_onnx_file_path=onnx_file,
        output_onnx_file_path=onnx_file,
        mode='full',
        search_mode='prefix_match',
        non_verbose=True,
    )
    rename(
        old_new=['/Mul','post_Mul'],
        input_onnx_file_path=onnx_file,
        output_onnx_file_path=onnx_file,
        mode='full',
        search_mode='prefix_match',
        non_verbose=True,
    )
    rename(
        old_new=['/Exp','post_Exp'],
        input_onnx_file_path=onnx_file,
        output_onnx_file_path=onnx_file,
        mode='full',
        search_mode='prefix_match',
        non_verbose=True,
    )
    rename(
        old_new=['/Sqrt','post_Sqrt'],
        input_onnx_file_path=onnx_file,
        output_onnx_file_path=onnx_file,
        mode='full',
        search_mode='prefix_match',
        non_verbose=True,
    )
    rename(
        old_new=['/Add','post_Add'],
        input_onnx_file_path=onnx_file,
        output_onnx_file_path=onnx_file,
        mode='full',
        search_mode='prefix_match',
        non_verbose=True,
    )
    rename(
        old_new=['/Sub','post_Sub'],
        input_onnx_file_path=onnx_file,
        output_onnx_file_path=onnx_file,
        mode='full',
        search_mode='prefix_match',
        non_verbose=True,
    )
    rename(
        old_new=['/Concat','post_Concat'],
        input_onnx_file_path=onnx_file,
        output_onnx_file_path=onnx_file,
        mode='full',
        search_mode='prefix_match',
        non_verbose=True,
    )
    rename(
        old_new=['/Constant','post_Constant'],
        input_onnx_file_path=onnx_file,
        output_onnx_file_path=onnx_file,
        mode='full',
        search_mode='prefix_match',
        non_verbose=True,
    )
