#! /usr/bin/env python

import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify
from argparse import ArgumentParser
import numpy as np
np.random.seed(0)
from sor4onnx import rename


class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

DTYPES_TO_TORCH_TYPES = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
    'bool': torch.bool,
}

class Model(nn.Module):
    def __init__(
        self,
        sub_a_constant,
    ):
        super(Model, self).__init__()
        self.sub_a_constant = sub_a_constant

    def forward(self, x):
        broadcast_sub = torch.subtract(torch.tensor(self.sub_a_constant), x)

        return broadcast_sub


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
        '-ms',
        '--model_name_suffix',
        type=int,
        default=0,
        help='Model name suffix',
    )
    parser.add_argument(
        '-p',
        '--persons',
        type=int,
        default=6,
        help='Number of person',
    )
    parser.add_argument(
        '-sac',
        '--sub_a_constat_path',
        type=str,
        required=True,
        help='constant file path'
    )
    parser.add_argument(
        '-sbs',
        '--sub_b_shape',
        type=int,
        nargs='+',
        required=True,
        help='SubB shape'
    )
    parser.add_argument(
        '-sbd',
        '--sub_b_data_type',
        type=str,
        required=True,
        default='float32',
        choices=DTYPES_TO_TORCH_TYPES.keys(),
        help='Type of SubB data',
    )
    args = parser.parse_args()

    opset = args.opset
    model_name_suffix = args.model_name_suffix
    sub_a_constat_path = args.sub_a_constat_path
    persons = args.persons
    sub_a_constant = np.load(sub_a_constat_path)
    sub_a_constant = np.tile(sub_a_constant, (1, 1, persons, 17))
    np.save(f'{sub_a_constat_path.split("_")[0]}x{persons}x17_{sub_a_constat_path.split("_")[1]}', sub_a_constant)

    sub_b_shape = args.sub_b_shape
    sub_b_data_type = args.sub_b_data_type

    MODEL = f'barracuda_broadcast_sub_{model_name_suffix}'

    model = Model(
        sub_a_constant,
    )
    onnx_file = f"{MODEL}_{sub_b_shape[0]*4}x{sub_b_shape[1]*4}_p{persons}.onnx"
    x = torch.randn(list(sub_b_shape), dtype=DTYPES_TO_TORCH_TYPES[sub_b_data_type])

    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=opset,
        input_names=[
            f'{MODEL}_input',
        ],
        output_names=[
            f'{MODEL}_output',
        ],
        do_constant_folding=False,
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    rename(
        old_new=["onnx::Sub_1", f"{MODEL}_a"],
        input_onnx_file_path=onnx_file,
        output_onnx_file_path=onnx_file,
    )
    rename(
        old_new=[f"{MODEL}_input", f"{MODEL}_b"],
        input_onnx_file_path=onnx_file,
        output_onnx_file_path=onnx_file,
    )