#!/bin/bash

python -m tf2onnx.convert \
--opset 11 \
--tflite face_blendshapes.tflite \
--output face_blendshapes.onnx \
--dequantize

onnxsim face_blendshapes.onnx face_blendshapes.onnx
onnxsim face_blendshapes.onnx face_blendshapes.onnx
onnxsim face_blendshapes.onnx face_blendshapes.onnx

sor4onnx \
--input_onnx_file_path face_blendshapes.onnx \
--old_new "serving_default_input_points:0" "input_points" \
--mode inputs \
--search_mode prefix_match \
--output_onnx_file_path face_blendshapes.onnx

sor4onnx \
--input_onnx_file_path face_blendshapes.onnx \
--old_new "StatefulPartitionedCall:0" "output" \
--mode outputs \
--search_mode prefix_match \
--output_onnx_file_path face_blendshapes.onnx





snd4onnx \
--remove_node_names model_1/GhumMarkerPoserMlpMixerGeneral/tf.__operators__.getitem/strided_slice1 \
--input_onnx_file_path face_blendshapes.onnx \
--output_onnx_file_path face_blendshapes.onnx


snd4onnx \
--remove_node_names model_1/tf.__operators__.getitem_2/strided_slice3 \
--input_onnx_file_path face_blendshapes.onnx \
--output_onnx_file_path face_blendshapes.onnx

snd4onnx \
--remove_node_names StatefulPartitionedCall:0 \
--input_onnx_file_path face_blendshapes.onnx \
--output_onnx_file_path face_blendshapes.onnx

snd4onnx \
--remove_node_names model_1/GhumMarkerPoserMlpMixerGeneral/reshape_7/Reshape \
--input_onnx_file_path face_blendshapes.onnx \
--output_onnx_file_path face_blendshapes.onnx


sor4onnx \
--input_onnx_file_path face_blendshapes.onnx \
--old_new "model_1/GhumMarkerPoserMlpMixerGeneral/" "" \
--mode full \
--search_mode partial_match \
--output_onnx_file_path face_blendshapes.onnx


sor4onnx \
--input_onnx_file_path face_blendshapes.onnx \
--old_new ";" "_" \
--mode full \
--search_mode partial_match \
--output_onnx_file_path face_blendshapes.onnx

sor4onnx \
--input_onnx_file_path face_blendshapes.onnx \
--old_new ":" "_" \
--mode full \
--search_mode partial_match \
--output_onnx_file_path face_blendshapes.onnx

sor4onnx \
--input_onnx_file_path face_blendshapes.onnx \
--old_new ":" "_" \
--mode full \
--search_mode partial_match \
--output_onnx_file_path face_blendshapes.onnx

sor4onnx \
--input_onnx_file_path face_blendshapes.onnx \
--old_new "MLPMixer/" "" \
--mode full \
--search_mode partial_match \
--output_onnx_file_path face_blendshapes.onnx



onnx2json \
--input_onnx_file_path face_blendshapes.onnx \
--output_json_path face_blendshapes.json \
--json_indent 2

json2onnx \
--input_json_path face_blendshapes.json \
--output_onnx_file_path face_blendshapes.onnx

onnxsim face_blendshapes.onnx face_blendshapes.onnx


onnx2tf -i face_blendshapes.onnx -osd -kat input_points -prf replace_face_blendshapes.json -coion -cotof
