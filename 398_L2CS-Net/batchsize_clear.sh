#!/bin/bash

sam4onnx \
--input_onnx_file_path l2cs_net_1x3x224x224.onnx \
--output_onnx_file_path l2cs_net_1x3x224x224.onnx \
--op_name /Reshape \
--input_constants /Constant_output_0 int64 [1,2048]

sam4onnx \
--input_onnx_file_path l2cs_net_1x3x336x336.onnx \
--output_onnx_file_path l2cs_net_1x3x336x336.onnx \
--op_name /Reshape \
--input_constants /Constant_output_0 int64 [1,2048]

sam4onnx \
--input_onnx_file_path l2cs_net_1x3x448x448.onnx \
--output_onnx_file_path l2cs_net_1x3x448x448.onnx \
--op_name /Reshape \
--input_constants /Constant_output_0 int64 [1,2048]

sam4onnx \
--input_onnx_file_path l2cs_net_1x3x560x560.onnx \
--output_onnx_file_path l2cs_net_1x3x560x560.onnx \
--op_name /Reshape \
--input_constants /Constant_output_0 int64 [1,2048]

sam4onnx \
--input_onnx_file_path l2cs_net_1x3x672x672.onnx \
--output_onnx_file_path l2cs_net_1x3x672x672.onnx \
--op_name /Reshape \
--input_constants /Constant_output_0 int64 [1,2048]



sbi4onnx \
--input_onnx_file_path l2cs_net_1x3x224x224.onnx \
--output_onnx_file_path l2cs_net_Nx3x224x224.onnx \
--initialization_character_string N

sbi4onnx \
--input_onnx_file_path l2cs_net_1x3x336x336.onnx \
--output_onnx_file_path l2cs_net_Nx3x336x336.onnx \
--initialization_character_string N

sbi4onnx \
--input_onnx_file_path l2cs_net_1x3x448x448.onnx \
--output_onnx_file_path l2cs_net_Nx3x448x448.onnx \
--initialization_character_string N

sbi4onnx \
--input_onnx_file_path l2cs_net_1x3x560x560.onnx \
--output_onnx_file_path l2cs_net_Nx3x560x560.onnx \
--initialization_character_string N

sbi4onnx \
--input_onnx_file_path l2cs_net_1x3x672x672.onnx \
--output_onnx_file_path l2cs_net_Nx3x672x672.onnx \
--initialization_character_string N



sam4onnx \
--input_onnx_file_path l2cs_net_Nx3x224x224.onnx \
--output_onnx_file_path l2cs_net_Nx3x224x224.onnx \
--op_name /Reshape \
--input_constants /Constant_output_0 int64 [-1,2048]

sam4onnx \
--input_onnx_file_path l2cs_net_Nx3x336x336.onnx \
--output_onnx_file_path l2cs_net_Nx3x336x336.onnx \
--op_name /Reshape \
--input_constants /Constant_output_0 int64 [-1,2048]

sam4onnx \
--input_onnx_file_path l2cs_net_Nx3x448x448.onnx \
--output_onnx_file_path l2cs_net_Nx3x448x448.onnx \
--op_name /Reshape \
--input_constants /Constant_output_0 int64 [-1,2048]

sam4onnx \
--input_onnx_file_path l2cs_net_Nx3x560x560.onnx \
--output_onnx_file_path l2cs_net_Nx3x560x560.onnx \
--op_name /Reshape \
--input_constants /Constant_output_0 int64 [-1,2048]

sam4onnx \
--input_onnx_file_path l2cs_net_Nx3x672x672.onnx \
--output_onnx_file_path l2cs_net_Nx3x672x672.onnx \
--op_name /Reshape \
--input_constants /Constant_output_0 int64 [-1,2048]

