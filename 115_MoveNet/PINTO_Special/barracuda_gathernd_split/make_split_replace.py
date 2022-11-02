#! /usr/bin/env python

import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
from argparse import ArgumentParser
import numpy as np
np.random.seed(0)
from ast import literal_eval


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
        split_axis,
        split_number_of_after_division,
    ):
        super(Model, self).__init__()
        self.split_axis = split_axis
        self.split_number_of_after_division = split_number_of_after_division

    def forward(self, x):
        result_splited_list = []
        split_start_idx_list = list(range(0, x.shape[self.split_axis], self.split_number_of_after_division))

        if self.split_axis == 0:
            for split_start_idx in split_start_idx_list:
                split_end_idx = split_start_idx + self.split_number_of_after_division
                result_splited_list.append(x[split_start_idx:split_end_idx, ...])

        elif self.split_axis == 1:
            for split_start_idx in split_start_idx_list:
                split_end_idx = split_start_idx + self.split_number_of_after_division
                result_splited_list.append(x[:, split_start_idx:split_end_idx, ...])

        elif self.split_axis == 2:
            for split_start_idx in split_start_idx_list:
                split_end_idx = split_start_idx + self.split_number_of_after_division
                result_splited_list.append(x[:, :, split_start_idx:split_end_idx, ...])

        elif self.split_axis == 3:
            for split_start_idx in split_start_idx_list:
                split_end_idx = split_start_idx + self.split_number_of_after_division
                result_splited_list.append(x[:, :, :, split_start_idx:split_end_idx, ...])

        elif self.split_axis == 4:
            for split_start_idx in split_start_idx_list:
                split_end_idx = split_start_idx + self.split_number_of_after_division
                result_splited_list.append(x[:, :, :, :, split_start_idx:split_end_idx, ...])

        elif self.split_axis == 5:
            for split_start_idx in split_start_idx_list:
                split_end_idx = split_start_idx + self.split_number_of_after_division
                result_splited_list.append(x[:, :, :, :, :, split_start_idx:split_end_idx, ...])

        elif self.split_axis == 6:
            for split_start_idx in split_start_idx_list:
                split_end_idx = split_start_idx + self.split_number_of_after_division
                result_splited_list.append(x[:, :, :, :, :, :, split_start_idx:split_end_idx, ...])

        elif self.split_axis == 7:
            for split_start_idx in split_start_idx_list:
                split_end_idx = split_start_idx + self.split_number_of_after_division
                result_splited_list.append(x[:, :, :, :, :, :, :, split_start_idx:split_end_idx, ...])

        elif self.split_axis == 8:
            for split_start_idx in split_start_idx_list:
                split_end_idx = split_start_idx + self.split_number_of_after_division
                result_splited_list.append(x[:, :, :, :, :, :, :, :, split_start_idx:split_end_idx, ...])

        elif self.split_axis == 9:
            for split_start_idx in split_start_idx_list:
                split_end_idx = split_start_idx + self.split_number_of_after_division
                result_splited_list.append(x[:, :, :, :, :, :, :, :, :, split_start_idx:split_end_idx, ...])

        elif self.split_axis == 10:
            for split_start_idx in split_start_idx_list:
                split_end_idx = split_start_idx + self.split_number_of_after_division
                result_splited_list.append(x[:, :, :, :, :, :, :, :, :, :, split_start_idx:split_end_idx])

        return result_splited_list


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
        '-ds',
        '--data_shape',
        type=str,
        nargs='+',
        required=True,
        help='Shape of input data',
    )
    parser.add_argument(
        '-dt',
        '--data_type',
        type=str,
        required=True,
        default='float32',
        choices=DTYPES_TO_TORCH_TYPES.keys(),
        help='Type of input data',
    )
    parser.add_argument(
        '-a',
        '--split_axis',
        type=int,
        default=-1,
        help='axis to split the input tensor'
    )
    parser.add_argument(
        '-n',
        '--split_number_of_after_division',
        type=int,
        default=1,
        help='\
            Number of elements after division.\n\
            e.g.\n\
            [1,16,8], n=1, split_axis=2 -> [1,16,1],[1,16,1],[1,16,1],[1,16,1],[1,16,1],[1,16,1],[1,16,1],[1,16,1]\n\
            [1,16,8], n=4, split_axis=2 -> [1,16,4],[1,16,4]'
    )

    args = parser.parse_args()

    opset=args.opset
    model_name_suffix = args.model_name_suffix

    data_shape = []
    for s in args.data_shape:
        try:
            val = literal_eval(s)
            if isinstance(val, int) and val >= 0:
                data_shape.append(val)
            else:
                data_shape.append(s)
        except:
            data_shape.append(s)
    data_shape = np.asarray(data_shape, dtype=np.int32)

    data_type = args.data_type
    split_axis = args.split_axis

    # split_axis check
    if split_axis > len(data_shape) - 1:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'split_axis must be less than or equal to the number of dimensions of data_shape. \n'+
            f'len(data_shape)-1: {len(data_shape) - 1} split_axis:{split_axis}'
        )
        sys.exit(1)

    if split_axis < 0 and abs(split_axis) > len(data_shape) - 1:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'split_axis must be less than or equal to the number of dimensions of data_shape. \n'+
            f'len(data_shape)-1: {len(data_shape) - 1} split_axis:{split_axis}'
        )
        sys.exit(1)

    if split_axis < 0:
        split_axis = len(data_shape) + split_axis

    if split_axis > 10:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'split_axis must specify 10 dimensions or less.'
        )
        sys.exit(1)

    split_number_of_after_division = args.split_number_of_after_division

    # split_number_of_after_division check
    if data_shape[split_axis] % split_number_of_after_division > 0:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The dimension to be divided must be divisible by split_number_of_after_division. \n'+
            f'data_shape[split_axis]: {data_shape[split_axis]} split_number_of_after_division: {split_number_of_after_division}'
        )
        sys.exit(1)

    split_number_of_groups = data_shape[split_axis] // split_number_of_after_division


    MODEL = f'barracuda_split_{model_name_suffix}'

    model = Model(
        split_axis,
        split_number_of_after_division,
    )
    onnx_file = f"{MODEL}.onnx"
    x = torch.randn(list(data_shape), dtype=DTYPES_TO_TORCH_TYPES[data_type])

    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=opset,
        input_names=[
            f'{MODEL}_input',
        ],
        output_names=[
            f'{MODEL}_split{num}_output' for num in range(split_number_of_groups)
        ],
        do_constant_folding=False,
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
