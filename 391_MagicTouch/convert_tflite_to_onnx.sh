#!/bin/bash

python -m tf2onnx.convert \
--opset 11 \
--tflite magic_touch.tflite \
--output magic_touch.onnx \
--inputs-as-nchw input_1 \
--dequantize

onnxsim magic_touch.onnx magic_touch.onnx
onnxsim magic_touch.onnx magic_touch.onnx
onnxsim magic_touch.onnx magic_touch.onnx

sor4onnx \
--input_onnx_file_path magic_touch.onnx \
--old_new "input_1" "input" \
--mode inputs \
--search_mode prefix_match \
--output_onnx_file_path magic_touch.onnx

sor4onnx \
--input_onnx_file_path magic_touch.onnx \
--old_new "Identity" "output" \
--mode outputs \
--search_mode prefix_match \
--output_onnx_file_path magic_touch.onnx

onnx2tf -i magic_touch.onnx -osd -coion -cotof
