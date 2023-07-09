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


####################################### Multi-batch

sne4onnx \
--input_onnx_file_path face_blendshapes.onnx \
--input_op_names input_points \
--output_op_names StatefulPartitionedCall_0_raw_output___3_0 \
--output_onnx_file_path face_blendshapes_split.onnx

sor4onnx \
--input_onnx_file_path reshape_11.onnx \
--old_new "/Reshape" "final_reshape" \
--mode full \
--search_mode prefix_match \
--output_onnx_file_path reshape_11.onnx

sor4onnx \
--input_onnx_file_path reshape_11.onnx \
--old_new "/Constant_output_0" "final_reshape_const" \
--mode full \
--search_mode prefix_match \
--output_onnx_file_path reshape_11.onnx

onnxsim reshape_11.onnx reshape_11.onnx

snc4onnx \
--input_onnx_file_paths face_blendshapes_split.onnx reshape_11.onnx \
--output_onnx_file_path face_blendshapes_batched.onnx \
--srcop_destop StatefulPartitionedCall_0_raw_output___3_0 input_reshape


onnxsim face_blendshapes_batched.onnx face_blendshapes_batched.onnx

####################################### N-batch

sbi4onnx \
--input_onnx_file_path face_blendshapes_batched.onnx \
--output_onnx_file_path face_blendshapes_Nx146x2.onnx \
--initialization_character_string N

onnx2json \
--input_onnx_file_path face_blendshapes_Nx146x2.onnx \
--output_json_path face_blendshapes_Nx146x2.json \
--json_indent 2

json2onnx \
--input_json_path face_blendshapes_Nx146x2.json \
--output_onnx_file_path face_blendshapes_Nx146x2.onnx




sor4onnx \
--input_onnx_file_path tile_11.onnx \
--old_new "/Shape" "tile_Shape" \
--mode full \
--search_mode exact_match \
--output_onnx_file_path tile_11.onnx

sor4onnx \
--input_onnx_file_path tile_11.onnx \
--old_new "/Gather" "tile_Gather" \
--mode full \
--search_mode exact_match \
--output_onnx_file_path tile_11.onnx

sor4onnx \
--input_onnx_file_path tile_11.onnx \
--old_new "/Unsqueeze" "tile_Unsqueeze" \
--mode full \
--search_mode exact_match \
--output_onnx_file_path tile_11.onnx

sor4onnx \
--input_onnx_file_path tile_11.onnx \
--old_new "/Concat" "tile_Concat" \
--mode full \
--search_mode exact_match \
--output_onnx_file_path tile_11.onnx

sor4onnx \
--input_onnx_file_path tile_11.onnx \
--old_new "/ConstantOfShape" "tile_ConstantOfShape" \
--mode full \
--search_mode exact_match \
--output_onnx_file_path tile_11.onnx

sor4onnx \
--input_onnx_file_path tile_11.onnx \
--old_new "/Mul" "tile_Mul" \
--mode full \
--search_mode exact_match \
--output_onnx_file_path tile_11.onnx

sor4onnx \
--input_onnx_file_path tile_11.onnx \
--old_new "/Concat_1" "tile_Concat_1" \
--mode full \
--search_mode exact_match \
--output_onnx_file_path tile_11.onnx

sor4onnx \
--input_onnx_file_path tile_11.onnx \
--old_new "/" "tile_" \
--mode full \
--search_mode prefix_match \
--output_onnx_file_path tile_11.onnx

onnxsim tile_11.onnx tile_11.onnx

onnx2json \
--input_onnx_file_path tile_11.onnx \
--output_json_path tile_11.json \
--json_indent 2

json2onnx \
--input_json_path tile_11.json \
--output_onnx_file_path tile_11.onnx


sne4onnx \
--input_onnx_file_path face_blendshapes_Nx146x2.onnx \
--input_op_names input_points \
--output_op_names input_tokens_embedding/BiasAdd_MixerBlock_3/mlp_channel_mixing/Mlp_2/Conv2D_input_tokens_embedding/Conv2D_input_tokens_embedding/BiasAdd/ReadVariableOp \
--output_onnx_file_path face_blendshapes_Nx146x2_1.onnx

snc4onnx \
--input_onnx_file_paths face_blendshapes_Nx146x2_1.onnx tile_11.onnx \
--output_onnx_file_path face_blendshapes_Nx146x2_2.onnx \
--srcop_destop input_tokens_embedding/BiasAdd_MixerBlock_3/mlp_channel_mixing/Mlp_2/Conv2D_input_tokens_embedding/Conv2D_input_tokens_embedding/BiasAdd/ReadVariableOp input_tile


sne4onnx \
--input_onnx_file_path face_blendshapes_Nx146x2.onnx \
--input_op_names AddExtraTokens/concat \
--output_op_names output \
--output_onnx_file_path face_blendshapes_Nx146x2_3.onnx

sbi4onnx \
--input_onnx_file_path face_blendshapes_Nx146x2_3.onnx \
--output_onnx_file_path face_blendshapes_Nx146x2_4.onnx \
--initialization_character_string N


snc4onnx \
--input_onnx_file_paths face_blendshapes_Nx146x2_2.onnx face_blendshapes_Nx146x2_4.onnx \
--output_onnx_file_path face_blendshapes_Nx146x2_5.onnx \
--srcop_destop output_tile AddExtraTokens/concat

