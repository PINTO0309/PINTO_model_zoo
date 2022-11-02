#! /usr/bin/env python

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from argparse import ArgumentParser

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()

    def forward(self, x):
        return x.to(torch.float32)

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()

    def forward(self, x):
        return x.to(torch.int32)


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
    args = parser.parse_args()

    MODEL = f'workaround_cast'
    OPSET=args.opset
    BATCHES = args.batches
    NUMPERSON = args.num_person


    # default: [1, 6, 1]
    x = torch.ones([BATCHES, NUMPERSON, 1], dtype=torch.int32)
    model1 = Model1()
    onnx_file = f"{MODEL}1_p{NUMPERSON}.onnx"
    torch.onnx.export(
        model1,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            'cast1_input',
        ],
        output_names=[
            'cast1_output',
        ],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)


    onnx_file = f"{MODEL}1_p{NUMPERSON}_N.onnx"
    torch.onnx.export(
        model1,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            'cast1_input',
        ],
        output_names=[
            'cast1_output',
        ],
        dynamic_axes={
            'cast1_input' : {0: 'batch'},
            'cast1_output': {0: 'batch'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)



    ##############################################

    x = torch.ones([BATCHES, NUMPERSON, 1], dtype=torch.float32)
    model2 = Model2()
    onnx_file = f"{MODEL}2_p{NUMPERSON}.onnx"
    torch.onnx.export(
        model2,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            'cast2_input',
        ],
        output_names=[
            'cast2_output',
        ],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)


    onnx_file = f"{MODEL}2_p{NUMPERSON}_N.onnx"
    torch.onnx.export(
        model2,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            'cast2_input',
        ],
        output_names=[
            'cast2_output',
        ],
        dynamic_axes={
            'cast2_input' : {0: 'batch'},
            'cast2_output': {0: 'batch'},
        }
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
