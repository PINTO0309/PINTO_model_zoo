#! /usr/bin/env python

import sys
import onnx
import onnx_graphsurgeon as gs
from typing import Optional
import struct
from argparse import ArgumentParser
from onnxsim import simplify

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


TARGET_INPUTS = [
    'predictions',
]

TARGET_VALUE_INFO = [
    'main01_boxes_cxcywh',
    'main01_onnx::Mul_15',
    'main01_onnx::Mul_10',

    'main01_onnx::Div_28',
    'main01_onnx::Add_30',
    'main01_onnx::Add_26',
    'main01_onnx::Unsqueeze_31',
    'main01_onnx::Concat_32',

    'main01_onnx::Div_20',
    'main01_onnx::Add_22',
    'main01_onnx::Add_18',
    'main01_onnx::Unsqueeze_23',
    'main01_onnx::Concat_24',

    'main01_onnx::Div_12',
    'main01_onnx::Sub_14',
    'main01_onnx::Sub_10',
    'main01_onnx::Unsqueeze_15',
    'main01_onnx::Concat_16',

    'main01_onnx::Div_4',
    'main01_onnx::Sub_6',
    'main01_onnx::Sub_2',
    'main01_onnx::Unsqueeze_7',
    'main01_onnx::Concat_8',

    'main01_y1x1y2x2',
    'main01_onnx::Transpose_16',
    'main01_scores',
]

def initialize(
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    output_onnx_file_path: Optional[str] = '',
    initialization_character_string: Optional[str] = 'batch',
    non_verbose: Optional[bool] = False,
) -> onnx.ModelProto:
    """
    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        Default: ''
    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.
    output_onnx_file_path: Optional[str]
        Output onnx file path. If not specified, no ONNX file is output.\n\
        Default: ''
    initialization_character_string: Optional[str]
        String to initialize batch size. "-1" or "N" or "xxx", etc...\n
        Default: 'batch'
    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False
    Returns
    -------
    changed_graph: onnx.ModelProto
        Changed onnx ModelProto.
    """

    # Unspecified check for input_onnx_file_path and onnx_graph
    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    if not initialization_character_string:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The initialization_character_string cannot be empty.'
        )
        sys.exit(1)

    # Loading Graphs
    # onnx_graph If specified, onnx_graph is processed first
    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)
    try:
        onnx_graph, _ = simplify(onnx_graph)
    except:
        pass
    graph = gs.import_onnx(onnx_graph)
    graph.cleanup().toposort()
    target_model = gs.export_onnx(graph)
    target_graph = target_model.graph

    for node in target_graph.input:
        if node.name in TARGET_INPUTS:
            if len(node.type.tensor_type.shape.dim)>0:
                node.type.tensor_type.shape.dim[0].dim_param = initialization_character_string

    target_value_info = [value_info for value_info in target_graph.value_info if value_info.name in TARGET_VALUE_INFO]
    for tensor in target_value_info:
        if len(tensor.type.tensor_type.shape.dim)>0:
            tensor.type.tensor_type.shape.dim[0].dim_param = initialization_character_string


    # infer_shapes
    target_model = onnx.shape_inference.infer_shapes(target_model)

    # Save
    if output_onnx_file_path:
        onnx.save(target_model, output_onnx_file_path)

    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')

    # Return
    return target_model


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-if',
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='Input onnx file path.'
    )
    parser.add_argument(
        '-of',
        '--output_onnx_file_path',
        type=str,
        required=True,
        help='Output onnx file path.'
    )
    parser.add_argument(
        '-ics',
        '--initialization_character_string',
        type=str,
        default='batch',
        help=\
            'String to initialize batch size. "-1" or "N" or "xxx", etc... \n'+
            'Default: \'batch\''
    )
    parser.add_argument(
        '-nv',
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    input_onnx_file_path = args.input_onnx_file_path
    output_onnx_file_path = args.output_onnx_file_path
    initialization_character_string = args.initialization_character_string
    non_verbose = args.non_verbose

    # Load
    onnx_graph = onnx.load(input_onnx_file_path)

    # Batchsize change
    changed_graph = initialize(
        input_onnx_file_path=None,
        onnx_graph=onnx_graph,
        output_onnx_file_path=output_onnx_file_path,
        initialization_character_string=initialization_character_string,
        non_verbose=non_verbose,
    )


if __name__ == '__main__':
    main()