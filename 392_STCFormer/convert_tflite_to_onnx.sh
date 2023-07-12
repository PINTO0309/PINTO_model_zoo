#!/bin/bash

onnx2tf -i stcformer_trans_1x27x17x2_with_prepost.onnx -kat input_n_f_k_xy -rtpo Erf -coion -cotof
onnx2tf -i stcformer_trans_1x81x17x2_with_prepost.onnx -kat input_n_f_k_xy -rtpo Erf -coion -cotof
