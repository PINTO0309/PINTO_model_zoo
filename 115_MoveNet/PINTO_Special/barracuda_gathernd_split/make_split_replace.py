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


"""
############################################################################################### 192x256, p6
############################################################################################### 192x256, p6
MODEL=barracuda_split_0
NUM=0
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_2" "${MODEL}_slice_starts${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_3" "${MODEL}_slice_ends${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_1" "${MODEL}_slice_axes${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_4" "${MODEL}_slice_steps${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "Slice_4" "${MODEL}_slice${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
NUM=1
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_7" "${MODEL}_slice_starts${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_8" "${MODEL}_slice_ends${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_6" "${MODEL}_slice_axes${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_9" "${MODEL}_slice_steps${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "Slice_9" "${MODEL}_slice${NUM}" \
--output_onnx_file_path ${MODEL}.onnx


MODEL=barracuda_split_1
NUM=0
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_2" "${MODEL}_slice_starts${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_3" "${MODEL}_slice_ends${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_1" "${MODEL}_slice_axes${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_4" "${MODEL}_slice_steps${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "Slice_4" "${MODEL}_slice${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
NUM=1
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_7" "${MODEL}_slice_starts${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_8" "${MODEL}_slice_ends${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_6" "${MODEL}_slice_axes${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_9" "${MODEL}_slice_steps${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "Slice_9" "${MODEL}_slice${NUM}" \
--output_onnx_file_path ${MODEL}.onnx


MODEL=barracuda_split_2
NUM=0
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_2" "${MODEL}_slice_starts${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_3" "${MODEL}_slice_ends${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_1" "${MODEL}_slice_axes${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_4" "${MODEL}_slice_steps${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "Slice_4" "${MODEL}_slice${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
NUM=1
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_7" "${MODEL}_slice_starts${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_8" "${MODEL}_slice_ends${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_6" "${MODEL}_slice_axes${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_9" "${MODEL}_slice_steps${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "Slice_9" "${MODEL}_slice${NUM}" \
--output_onnx_file_path ${MODEL}.onnx


MODEL=barracuda_split_3
NUM=0
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_2" "${MODEL}_slice_starts${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_3" "${MODEL}_slice_ends${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_4" "${MODEL}_slice_steps${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "Slice_4" "${MODEL}_slice${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
NUM=1
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_7" "${MODEL}_slice_starts${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_8" "${MODEL}_slice_ends${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_6" "${MODEL}_slice_axes${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_9" "${MODEL}_slice_steps${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "Slice_9" "${MODEL}_slice${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
NUM=2
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_12" "${MODEL}_slice_starts${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_13" "${MODEL}_slice_ends${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_11" "${MODEL}_slice_axes${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_14" "${MODEL}_slice_steps${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "Slice_14" "${MODEL}_slice${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
NUM=3
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_17" "${MODEL}_slice_starts${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_18" "${MODEL}_slice_ends${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_16" "${MODEL}_slice_axes${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_19" "${MODEL}_slice_steps${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "Slice_19" "${MODEL}_slice${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
NUM=0
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_1" "${MODEL}_slice_axes${NUM}" \
--output_onnx_file_path ${MODEL}.onnx


MODEL=barracuda_split_4
NUM=0
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_2" "${MODEL}_slice_starts${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_3" "${MODEL}_slice_ends${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_1" "${MODEL}_slice_axes${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_4" "${MODEL}_slice_steps${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "Slice_4" "${MODEL}_slice${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
NUM=1
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_7" "${MODEL}_slice_starts${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_8" "${MODEL}_slice_ends${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_6" "${MODEL}_slice_axes${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "onnx::Slice_9" "${MODEL}_slice_steps${NUM}" \
--output_onnx_file_path ${MODEL}.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "Slice_9" "${MODEL}_slice${NUM}" \
--output_onnx_file_path ${MODEL}.onnx

################################################################################### Merge Process
MODEL=movenet_multipose_lightning_192x256_p10_nopost_myriad
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521" "gnd0_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd__522" "gnd01_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536" "gnd1_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx

sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442" "gnd2_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd_2__495" "gnd2_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx

sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/Reshape_8" "gnd3_Reshape" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd_4__593" "gnd34_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/Max" "gnd4_ReduceMax" \
--output_onnx_file_path ${MODEL}_barracuda.onnx


###################################################################################
MODEL2=${MODEL}_barracuda
NUM=0
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd0_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=1
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd1_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_1 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=2
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd2_Transpose bgn${NUM}_data gnd2_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=3
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd3_Reshape bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=4
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd4_ReduceMax bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx


###################################################################################
MODEL2=${MODEL}_barracuda
NUM=0
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop Max__524 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=1
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Reshape_7 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_1" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_1:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_1 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=2
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Reshape_9 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_2" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_2:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_2 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=3
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Squeeze_4 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/split" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/split:1" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split2_output" \
--to_input_variable_name "StatefulPartitionedCall/split:2" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split3_output" \
--to_input_variable_name "StatefulPartitionedCall/split:3" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/split \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=4
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop bgn3_output barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_3" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_3:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_3 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx


############################################################################################### 192x256, p10
############################################################################################### 192x256, p10
NUM=0
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "6"/"dimValue": "10"/g' barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=1
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "6"/"dimValue": "10"/g' barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=2
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "6"/"dimValue": "10"/g' barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=3
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "102"/"dimValue": "170"/g' barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=4
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "102"/"dimValue": "170"/g' barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json




NUM=0
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "6"/"dimValue": "10"/g' barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json

NUM=1
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "6"/"dimValue": "10"/g' barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json

NUM=2
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "6"/"dimValue": "10"/g' barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json

NUM=3
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "6"/"dimValue": "10"/g' barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json

NUM=4
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "102"/"dimValue": "170"/g' barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json


################################################################################### Merge Process
MODEL=movenet_multipose_lightning_192x256_p10_nopost_myriad
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521" "gnd0_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd__522" "gnd01_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536" "gnd1_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx

sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442" "gnd2_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd_2__495" "gnd2_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx

sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/Reshape_8" "gnd3_Reshape" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd_4__593" "gnd34_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/Max" "gnd4_ReduceMax" \
--output_onnx_file_path ${MODEL}_barracuda.onnx


###################################################################################
MODEL2=${MODEL}_barracuda
NUM=0
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd0_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=1
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd1_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_1 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=2
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd2_Transpose bgn${NUM}_data gnd2_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=3
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd3_Reshape bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=4
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd4_ReduceMax bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx


###################################################################################
MODEL2=${MODEL}_barracuda
NUM=0
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop Max__524 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=1
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Reshape_7 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_1" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_1:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_1 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=2
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Reshape_9 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_2" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_2:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_2 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=3
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Squeeze_4 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/split" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/split:1" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split2_output" \
--to_input_variable_name "StatefulPartitionedCall/split:2" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split3_output" \
--to_input_variable_name "StatefulPartitionedCall/split:3" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/split \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=4
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop bgn3_output barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_3" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_3:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_3 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx


############################################################################################### 192x256, p20
############################################################################################### 192x256, p20

############################################################################################### GatherND
NUM=0
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "10"/"dimValue": "20"/g' barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=1
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "10"/"dimValue": "20"/g' barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=2
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "10"/"dimValue": "20"/g' barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=3
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "170"/"dimValue": "340"/g' barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=4
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "170"/"dimValue": "340"/g' barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

############################################################################################### Split
NUM=0
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "10"/"dimValue": "20"/g' barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json

NUM=1
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "10"/"dimValue": "20"/g' barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json

NUM=2
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "10"/"dimValue": "20"/g' barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json

NUM=3
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "10"/"dimValue": "20"/g' barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json

NUM=4
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e 's/"dimValue": "170"/"dimValue": "340"/g' barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json


################################################################################### Merge Process
MODEL=movenet_multipose_lightning_192x256_p20_nopost_myriad
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521" "gnd0_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd__522" "gnd01_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536" "gnd1_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx

sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442" "gnd2_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd_2__495" "gnd2_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx

sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/Reshape_8" "gnd3_Reshape" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd_4__593" "gnd34_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/Max" "gnd4_ReduceMax" \
--output_onnx_file_path ${MODEL}_barracuda.onnx


###################################################################################
MODEL2=${MODEL}_barracuda
NUM=0
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd0_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=1
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd1_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_1 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=2
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd2_Transpose bgn${NUM}_data gnd2_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=3
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd3_Reshape bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=4
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd4_ReduceMax bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx


###################################################################################
MODEL2=${MODEL}_barracuda
NUM=0
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop Max__524 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=1
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Reshape_7 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_1" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_1:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_1 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=2
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Reshape_9 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_2" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_2:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_2 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=3
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Squeeze_4 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/split" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/split:1" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split2_output" \
--to_input_variable_name "StatefulPartitionedCall/split:2" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split3_output" \
--to_input_variable_name "StatefulPartitionedCall/split:3" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/split \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=4
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop bgn3_output barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_3" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_3:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_3 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx



############################################################################################### 192x320, p6
############################################################################################### 192x320, p6
cp 192x256_p6/* .

############################################################################################### GatherND
H=192
W=320
P=6
QUATH=$((${H} / 4))
QUATW=$((${W} / 4))
NEWSHAPE1=$((${QUATH} * ${QUATW}))
NEWSHAPE2=$((${QUATH} * ${QUATW} * 17))
NUM=0
MULVAL0=`sed4onnx --constant_string [${NEWSHAPE1},${QUATW},1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"AAwAAAAAAABAAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL0}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=1
MULVAL1=`sed4onnx --constant_string [${NEWSHAPE1},${QUATW},1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"AAwAAAAAAABAAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL1}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=2
MULVAL2=`sed4onnx --constant_string [${QUATW},1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"QAAAAAAAAAABAAAAAAAAAA==\"/\"rawData\": \"${MULVAL2}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=3
MULVAL3=`sed4onnx --constant_string [$((${NEWSHAPE2} / ${QUATH})),17,1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"52224\"/\"dimValue\": \"${NEWSHAPE2}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"QAQAAAAAAAARAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL3}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=4
MULVAL4=`sed4onnx --constant_string [$((${NEWSHAPE2} / ${QUATH})),17,1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"52224\"/\"dimValue\": \"${NEWSHAPE2}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"QAQAAAAAAAARAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL4}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json


############################################################################################### Split
NUM=0
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json
NUM=1
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json
NUM=2
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json
NUM=3
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json
NUM=4
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json

############################################################################################### Merge Process
MODEL=movenet_multipose_lightning_${H}x${W}_p${P}_nopost_myriad
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521" "gnd0_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd__522" "gnd01_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536" "gnd1_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx

sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442" "gnd2_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd_2__495" "gnd2_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx

sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/Reshape_8" "gnd3_Reshape" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd_4__593" "gnd34_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/Max" "gnd4_ReduceMax" \
--output_onnx_file_path ${MODEL}_barracuda.onnx


###################################################################################
MODEL2=${MODEL}_barracuda
NUM=0
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd0_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=1
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd1_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_1 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=2
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd2_Transpose bgn${NUM}_data gnd2_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=3
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd3_Reshape bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=4
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd4_ReduceMax bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx


###################################################################################
MODEL2=${MODEL}_barracuda
NUM=0
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop Max__524 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=1
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Reshape_7 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_1" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_1:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_1 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=2
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Reshape_9 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_2" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_2:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_2 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=3
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Squeeze_4 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/split" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/split:1" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split2_output" \
--to_input_variable_name "StatefulPartitionedCall/split:2" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split3_output" \
--to_input_variable_name "StatefulPartitionedCall/split:3" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/split \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=4
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop bgn3_output barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_3" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_3:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_3 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

###################################################################################
mkdir -p ${H}x${W}_p${P}
mv barracuda_gather_nd_*.onnx ${H}x${W}_p${P}
mv barracuda_split_*.onnx ${H}x${W}_p${P}


############################################################################################### 192x320, p10
############################################################################################### 192x320, p10
cp 192x256_p6/* .

############################################################################################### GatherND
H=192
W=320
P=10
QUATH=$((${H} / 4))
QUATW=$((${W} / 4))
NEWSHAPE1=$((${QUATH} * ${QUATW}))
NEWSHAPE2=$((${QUATH} * ${QUATW} * 17))
NUM=0
MULVAL0=`sed4onnx --constant_string [${NEWSHAPE1},${QUATW},1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"AAwAAAAAAABAAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL0}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=1
MULVAL1=`sed4onnx --constant_string [${NEWSHAPE1},${QUATW},1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"AAwAAAAAAABAAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL1}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=2
MULVAL2=`sed4onnx --constant_string [${QUATW},1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"QAAAAAAAAAABAAAAAAAAAA==\"/\"rawData\": \"${MULVAL2}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=3
MULVAL3=`sed4onnx --constant_string [$((${NEWSHAPE2} / ${QUATH})),17,1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"52224\"/\"dimValue\": \"${NEWSHAPE2}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"QAQAAAAAAAARAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL3}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=4
MULVAL4=`sed4onnx --constant_string [$((${NEWSHAPE2} / ${QUATH})),17,1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"52224\"/\"dimValue\": \"${NEWSHAPE2}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"QAQAAAAAAAARAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL4}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json


############################################################################################### Split
NUM=0
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json
NUM=1
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json
NUM=2
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json
NUM=3
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json
NUM=4
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json

############################################################################################### Merge Process
MODEL=movenet_multipose_lightning_${H}x${W}_p${P}_nopost_myriad
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521" "gnd0_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd__522" "gnd01_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536" "gnd1_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx

sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442" "gnd2_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd_2__495" "gnd2_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx

sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/Reshape_8" "gnd3_Reshape" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd_4__593" "gnd34_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/Max" "gnd4_ReduceMax" \
--output_onnx_file_path ${MODEL}_barracuda.onnx


###################################################################################
MODEL2=${MODEL}_barracuda
NUM=0
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd0_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=1
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd1_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_1 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=2
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd2_Transpose bgn${NUM}_data gnd2_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=3
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd3_Reshape bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=4
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd4_ReduceMax bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx


###################################################################################
MODEL2=${MODEL}_barracuda
NUM=0
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop Max__524 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=1
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Reshape_7 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_1" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_1:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_1 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=2
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Reshape_9 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_2" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_2:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_2 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=3
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Squeeze_4 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/split" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/split:1" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split2_output" \
--to_input_variable_name "StatefulPartitionedCall/split:2" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split3_output" \
--to_input_variable_name "StatefulPartitionedCall/split:3" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/split \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=4
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop bgn3_output barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_3" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_3:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_3 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

###################################################################################
mkdir -p ${H}x${W}_p${P}
mv barracuda_gather_nd_*.onnx ${H}x${W}_p${P}
mv barracuda_split_*.onnx ${H}x${W}_p${P}


############################################################################################### 192x320, p20
############################################################################################### 192x320, p20
cp 192x256_p6/* .

############################################################################################### GatherND
H=192
W=320
P=20
QUATH=$((${H} / 4))
QUATW=$((${W} / 4))
NEWSHAPE1=$((${QUATH} * ${QUATW}))
NEWSHAPE2=$((${QUATH} * ${QUATW} * 17))
NUM=0
MULVAL0=`sed4onnx --constant_string [${NEWSHAPE1},${QUATW},1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"AAwAAAAAAABAAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL0}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=1
MULVAL1=`sed4onnx --constant_string [${NEWSHAPE1},${QUATW},1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"AAwAAAAAAABAAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL1}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=2
MULVAL2=`sed4onnx --constant_string [${QUATW},1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"QAAAAAAAAAABAAAAAAAAAA==\"/\"rawData\": \"${MULVAL2}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=3
MULVAL3=`sed4onnx --constant_string [$((${NEWSHAPE2} / ${QUATH})),17,1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"52224\"/\"dimValue\": \"${NEWSHAPE2}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"QAQAAAAAAAARAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL3}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json

NUM=4
MULVAL4=`sed4onnx --constant_string [$((${NEWSHAPE2} / ${QUATH})),17,1] --dtype int64 --mode encode`
onnx2json \
--input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
--output_json_path barracuda_gather_nd_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"dimValue\": \"52224\"/\"dimValue\": \"${NEWSHAPE2}\"/g" barracuda_gather_nd_${NUM}.json
sed -i -e "s/\"rawData\": \"QAQAAAAAAAARAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL4}\"/g" barracuda_gather_nd_${NUM}.json
json2onnx \
--input_json_path barracuda_gather_nd_${NUM}.json \
--output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
rm barracuda_gather_nd_${NUM}.json


############################################################################################### Split
NUM=0
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json
NUM=1
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json
NUM=2
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json
NUM=3
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json
NUM=4
onnx2json \
--input_onnx_file_path barracuda_split_${NUM}.onnx \
--output_json_path barracuda_split_${NUM}.json \
--json_indent 2
sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_split_${NUM}.json
json2onnx \
--input_json_path barracuda_split_${NUM}.json \
--output_onnx_file_path barracuda_split_${NUM}.onnx
rm barracuda_split_${NUM}.json

############################################################################################### Merge Process
MODEL=movenet_multipose_lightning_${H}x${W}_p${P}_nopost_myriad
sor4onnx \
--input_onnx_file_path ${MODEL}.onnx \
--old_new "StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521" "gnd0_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd__522" "gnd01_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536" "gnd1_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx

sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442" "gnd2_Transpose" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd_2__495" "gnd2_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx

sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/Reshape_8" "gnd3_Reshape" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/GatherNd_4__593" "gnd34_Cast" \
--output_onnx_file_path ${MODEL}_barracuda.onnx
sor4onnx \
--input_onnx_file_path ${MODEL}_barracuda.onnx \
--old_new "StatefulPartitionedCall/Max" "gnd4_ReduceMax" \
--output_onnx_file_path ${MODEL}_barracuda.onnx


###################################################################################
MODEL2=${MODEL}_barracuda
NUM=0
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd0_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=1
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd1_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_1 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=2
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd2_Transpose bgn${NUM}_data gnd2_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=3
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd3_Reshape bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=4
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
--srcop_destop gnd4_ReduceMax bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "bgn${NUM}_output" \
--to_input_variable_name "StatefulPartitionedCall/GatherNd_${NUM}" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/GatherNd_${NUM} \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx


###################################################################################
MODEL2=${MODEL}_barracuda
NUM=0
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop Max__524 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=1
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Reshape_7 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_1" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_1:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_1 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=2
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Reshape_9 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_2" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_2:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_2 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=3
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop StatefulPartitionedCall/Squeeze_4 barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/split" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/split:1" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split2_output" \
--to_input_variable_name "StatefulPartitionedCall/split:2" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split3_output" \
--to_input_variable_name "StatefulPartitionedCall/split:3" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/split \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

NUM=4
snc4onnx \
--input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
--srcop_destop bgn3_output barracuda_split_${NUM}_input \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split0_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_3" \
--output_onnx_file_path ${MODEL2}.onnx
svs4onnx \
--input_onnx_file_path ${MODEL2}.onnx \
--from_output_variable_name "barracuda_split_${NUM}_split1_output" \
--to_input_variable_name "StatefulPartitionedCall/unstack_3:1" \
--output_onnx_file_path ${MODEL2}.onnx
snd4onnx \
--remove_node_names StatefulPartitionedCall/unstack_3 \
--input_onnx_file_path ${MODEL2}.onnx \
--output_onnx_file_path ${MODEL2}.onnx

###################################################################################
mkdir -p ${H}x${W}_p${P}
mv barracuda_gather_nd_*.onnx ${H}x${W}_p${P}
mv barracuda_split_*.onnx ${H}x${W}_p${P}

"""




















"""
0 gathernd0_transpose
1 gathernd0_cast
2 gathernd0_reshape
3 gathernd1_transpose
4 gathernd1_cast
5 gathernd1_reshape
6 gathernd2_transpose
7 gathernd2_cast
8 gathernd2_reshape
9 gathernd3_reshape
10 gathernd3_cast
11 gathernd3_split
12 gathernd4_reducemax
13 gathernd4_cast
14 gathernd4_reshape

GATHERND0_TRANSPOSE
GATHERND0_CAST
GATHERND0_RESHAPE
GATHERND1_TRANSPOSE
GATHERND1_CAST
GATHERND1_RESHAPE
GATHERND2_TRANSPOSE
GATHERND2_CAST
GATHERND2_RESHAPE
GATHERND3_RESHAPE
GATHERND3_CAST
GATHERND3_SPLIT
GATHERND4_REDUCEMAX
GATHERND4_CAST
GATHERND4_RESHAPE



PARAMLIST=(
    "192 256 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "192 256 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/stack_1_Concat__512 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "192 256 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"

    "192 320 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "192 320 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "192 320 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"

    "256 320 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "256 320 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "256 320 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"

    "256 416 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "256 416 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "256 416 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"

    "288 480 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "288 480 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "288 480 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"

    "384 640 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "384 640 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "384 640 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"

    "384 1280 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "384 1280 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "384 1280 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"

    "480 640 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "480 640 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "480 640 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"

    "480 800 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "480 800 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "480 800 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"

    "736 1280 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "736 1280 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "736 1280 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
)


PARAMLIST=(
    "192 256 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "192 256 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/stack_1_Concat__512 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "192 256 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
)
for((i=0; i<${#PARAMLIST[@]}; i++))
do
    PARAM=(`echo ${PARAMLIST[i]}`)
    H=${PARAM[0]}
    W=${PARAM[1]}
    P=${PARAM[2]}

    GATHERND0_TRANSPOSE=${PARAM[3]}
    GATHERND0_CAST=${PARAM[4]}
    GATHERND0_RESHAPE=${PARAM[5]}
    GATHERND1_TRANSPOSE=${PARAM[6]}
    GATHERND1_CAST=${PARAM[7]}
    GATHERND1_RESHAPE=${PARAM[8]}
    GATHERND2_TRANSPOSE=${PARAM[9]}
    GATHERND2_CAST=${PARAM[10]}
    GATHERND2_RESHAPE=${PARAM[11]}
    GATHERND3_RESHAPE=${PARAM[12]}
    GATHERND3_CAST=${PARAM[13]}
    GATHERND3_SPLIT=${PARAM[14]}
    GATHERND4_REDUCEMAX=${PARAM[15]}
    GATHERND4_CAST=${PARAM[16]}
    GATHERND4_RESHAPE=${PARAM[17]}

    echo @@@@@@@@@@@@@@@@@ processing ${H}x${W} p=${P} ...

    cp 192x256_p6/* .

    ############################################################################################### GatherND
    QUATH=$((${H} / 4))
    QUATW=$((${W} / 4))
    NEWSHAPE1=$((${QUATH} * ${QUATW}))
    NEWSHAPE2=$((${QUATH} * ${QUATW} * 17))
    NUM=0
    MULVAL0=`sed4onnx --constant_string [${NEWSHAPE1},${QUATW},1] --dtype int64 --mode encode`
    onnx2json \
    --input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
    --output_json_path barracuda_gather_nd_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"rawData\": \"AAwAAAAAAABAAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL0}\"/g" barracuda_gather_nd_${NUM}.json
    json2onnx \
    --input_json_path barracuda_gather_nd_${NUM}.json \
    --output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
    rm barracuda_gather_nd_${NUM}.json

    NUM=1
    MULVAL1=`sed4onnx --constant_string [${NEWSHAPE1},${QUATW},1] --dtype int64 --mode encode`
    onnx2json \
    --input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
    --output_json_path barracuda_gather_nd_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"rawData\": \"AAwAAAAAAABAAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL1}\"/g" barracuda_gather_nd_${NUM}.json
    json2onnx \
    --input_json_path barracuda_gather_nd_${NUM}.json \
    --output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
    rm barracuda_gather_nd_${NUM}.json

    NUM=2
    MULVAL2=`sed4onnx --constant_string [${QUATW},1] --dtype int64 --mode encode`
    onnx2json \
    --input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
    --output_json_path barracuda_gather_nd_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"rawData\": \"QAAAAAAAAAABAAAAAAAAAA==\"/\"rawData\": \"${MULVAL2}\"/g" barracuda_gather_nd_${NUM}.json
    json2onnx \
    --input_json_path barracuda_gather_nd_${NUM}.json \
    --output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
    rm barracuda_gather_nd_${NUM}.json

    NUM=3
    MULVAL3=`sed4onnx --constant_string [$((${NEWSHAPE2} / ${QUATH})),17,1] --dtype int64 --mode encode`
    onnx2json \
    --input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
    --output_json_path barracuda_gather_nd_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"52224\"/\"dimValue\": \"${NEWSHAPE2}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"rawData\": \"QAQAAAAAAAARAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL3}\"/g" barracuda_gather_nd_${NUM}.json
    json2onnx \
    --input_json_path barracuda_gather_nd_${NUM}.json \
    --output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
    rm barracuda_gather_nd_${NUM}.json

    NUM=4
    MULVAL4=`sed4onnx --constant_string [$((${NEWSHAPE2} / ${QUATH})),17,1] --dtype int64 --mode encode`
    onnx2json \
    --input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
    --output_json_path barracuda_gather_nd_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"52224\"/\"dimValue\": \"${NEWSHAPE2}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"rawData\": \"QAQAAAAAAAARAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL4}\"/g" barracuda_gather_nd_${NUM}.json
    json2onnx \
    --input_json_path barracuda_gather_nd_${NUM}.json \
    --output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
    rm barracuda_gather_nd_${NUM}.json


    ############################################################################################### Split
    NUM=0
    onnx2json \
    --input_onnx_file_path barracuda_split_${NUM}.onnx \
    --output_json_path barracuda_split_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
    json2onnx \
    --input_json_path barracuda_split_${NUM}.json \
    --output_onnx_file_path barracuda_split_${NUM}.onnx
    rm barracuda_split_${NUM}.json
    NUM=1
    onnx2json \
    --input_onnx_file_path barracuda_split_${NUM}.onnx \
    --output_json_path barracuda_split_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
    json2onnx \
    --input_json_path barracuda_split_${NUM}.json \
    --output_onnx_file_path barracuda_split_${NUM}.onnx
    rm barracuda_split_${NUM}.json
    NUM=2
    onnx2json \
    --input_onnx_file_path barracuda_split_${NUM}.onnx \
    --output_json_path barracuda_split_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
    json2onnx \
    --input_json_path barracuda_split_${NUM}.json \
    --output_onnx_file_path barracuda_split_${NUM}.onnx
    rm barracuda_split_${NUM}.json
    NUM=3
    onnx2json \
    --input_onnx_file_path barracuda_split_${NUM}.onnx \
    --output_json_path barracuda_split_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
    json2onnx \
    --input_json_path barracuda_split_${NUM}.json \
    --output_onnx_file_path barracuda_split_${NUM}.onnx
    rm barracuda_split_${NUM}.json
    NUM=4
    onnx2json \
    --input_onnx_file_path barracuda_split_${NUM}.onnx \
    --output_json_path barracuda_split_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_split_${NUM}.json
    json2onnx \
    --input_json_path barracuda_split_${NUM}.json \
    --output_onnx_file_path barracuda_split_${NUM}.onnx
    rm barracuda_split_${NUM}.json

    ############################################################################################### Merge Process
    MODEL=movenet_multipose_lightning_${H}x${W}_p${P}_nopost_myriad
    echo 001
    sor4onnx \
    --input_onnx_file_path ${MODEL}.onnx \
    --old_new "${GATHERND0_TRANSPOSE}" "gnd0_Transpose" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx
    echo 002
    sor4onnx \
    --input_onnx_file_path ${MODEL}_barracuda.onnx \
    --old_new "${GATHERND0_CAST}" "gnd01_Cast" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx
    echo 003
    sor4onnx \
    --input_onnx_file_path ${MODEL}_barracuda.onnx \
    --old_new "${GATHERND1_TRANSPOSE}" "gnd1_Transpose" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx

    echo 004
    sor4onnx \
    --input_onnx_file_path ${MODEL}_barracuda.onnx \
    --old_new "${GATHERND2_TRANSPOSE}" "gnd2_Transpose" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx
    echo 005
    sor4onnx \
    --input_onnx_file_path ${MODEL}_barracuda.onnx \
    --old_new "${GATHERND2_CAST}" "gnd2_Cast" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx

    echo 006
    sor4onnx \
    --input_onnx_file_path ${MODEL}_barracuda.onnx \
    --old_new "${GATHERND3_RESHAPE}" "gnd3_Reshape" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx
    echo 007
    sor4onnx \
    --input_onnx_file_path ${MODEL}_barracuda.onnx \
    --old_new "${GATHERND3_CAST}" "gnd34_Cast" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx
    echo 008
    sor4onnx \
    --input_onnx_file_path ${MODEL}_barracuda.onnx \
    --old_new "${GATHERND4_REDUCEMAX}" "gnd4_ReduceMax" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx


    ###################################################################################
    MODEL2=${MODEL}_barracuda
    NUM=0
    echo 009
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
    --srcop_destop gnd0_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 010
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "bgn${NUM}_output" \
    --to_input_variable_name "${GATHERND0_RESHAPE}" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 011
    snd4onnx \
    --remove_node_names ${GATHERND0_RESHAPE} \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=1
    echo 012
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
    --srcop_destop gnd1_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 013
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "bgn${NUM}_output" \
    --to_input_variable_name "${GATHERND1_RESHAPE}" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 014
    snd4onnx \
    --remove_node_names ${GATHERND1_RESHAPE} \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=2
    echo 015
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
    --srcop_destop gnd2_Transpose bgn${NUM}_data gnd2_Cast bgn${NUM}_indices \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 016
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "bgn${NUM}_output" \
    --to_input_variable_name "${GATHERND2_RESHAPE}" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 017
    snd4onnx \
    --remove_node_names ${GATHERND2_RESHAPE} \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=3
    echo 018
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
    --srcop_destop gnd3_Reshape bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 019
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "bgn${NUM}_output" \
    --to_input_variable_name "${GATHERND3_SPLIT}" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 020
    snd4onnx \
    --remove_node_names ${GATHERND3_SPLIT} \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=4
    echo 021
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
    --srcop_destop gnd4_ReduceMax bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 022
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "bgn${NUM}_output" \
    --to_input_variable_name "${GATHERND4_RESHAPE}" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 023
    snd4onnx \
    --remove_node_names ${GATHERND4_RESHAPE} \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx


    ###################################################################################
    MODEL2=${MODEL}_barracuda
    NUM=0
    echo 024
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
    --srcop_destop Max__524 barracuda_split_${NUM}_input \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 025
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split0_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 026
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split1_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack:1" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 027
    snd4onnx \
    --remove_node_names StatefulPartitionedCall/unstack \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=1
    echo 028
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
    --srcop_destop StatefulPartitionedCall/Reshape_7 barracuda_split_${NUM}_input \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 029
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split0_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack_1" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 030
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split1_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack_1:1" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 031
    snd4onnx \
    --remove_node_names StatefulPartitionedCall/unstack_1 \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=2
    echo 032
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
    --srcop_destop StatefulPartitionedCall/Reshape_9 barracuda_split_${NUM}_input \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 033
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split0_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack_2" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 034
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split1_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack_2:1" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 035
    snd4onnx \
    --remove_node_names StatefulPartitionedCall/unstack_2 \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=3
    echo 036
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
    --srcop_destop StatefulPartitionedCall/Squeeze_4 barracuda_split_${NUM}_input \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 037
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split0_output" \
    --to_input_variable_name "StatefulPartitionedCall/split" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 038
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split1_output" \
    --to_input_variable_name "StatefulPartitionedCall/split:1" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 039
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split2_output" \
    --to_input_variable_name "StatefulPartitionedCall/split:2" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 040
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split3_output" \
    --to_input_variable_name "StatefulPartitionedCall/split:3" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 041
    snd4onnx \
    --remove_node_names StatefulPartitionedCall/split \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=4
    echo 042
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
    --srcop_destop bgn3_output barracuda_split_${NUM}_input \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 043
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split0_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack_3" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 044
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split1_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack_3:1" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 045
    snd4onnx \
    --remove_node_names StatefulPartitionedCall/unstack_3 \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    ###################################################################################
    mkdir -p ${H}x${W}_p${P}
    mv barracuda_gather_nd_*.onnx ${H}x${W}_p${P}
    mv barracuda_split_*.onnx ${H}x${W}_p${P}
done
"""