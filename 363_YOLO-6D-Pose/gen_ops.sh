OPSET=11

sog4onnx \
--op_type Constant \
--opset ${OPSET} \
--op_name const_op_zero \
--output_variables const_zero_var int64 [1] \
--attributes value int64 [0] \
--output_onnx_file_path const_op_zero.onnx \
--non_verbose

sog4onnx \
--op_type Constant \
--opset ${OPSET} \
--op_name const_op_one_1 \
--output_variables const_one_var_1 int64 [1] \
--attributes value int64 [1] \
--output_onnx_file_path const_op_one_1.onnx \
--non_verbose

sog4onnx \
--op_type Constant \
--opset ${OPSET} \
--op_name const_op_one_2 \
--output_variables const_one_var_2 int64 [1] \
--attributes value int64 [1] \
--output_onnx_file_path const_op_one_2.onnx \
--non_verbose

sog4onnx \
--op_type Constant \
--opset ${OPSET} \
--op_name const_op_two \
--output_variables const_two_var int64 [1] \
--attributes value int64 [2] \
--output_onnx_file_path const_op_two.onnx \
--non_verbose

sog4onnx \
--op_type Constant \
--opset ${OPSET} \
--op_name const_op_four \
--output_variables const_four_var int64 [1] \
--attributes value int64 [4] \
--output_onnx_file_path const_op_four.onnx \
--non_verbose

sog4onnx \
--op_type Constant \
--opset ${OPSET} \
--op_name const_op_twenty_five \
--output_variables const_twenty_five_var int64 [1] \
--attributes value int64 [25] \
--output_onnx_file_path const_op_twenty_five.onnx \
--non_verbose

sog4onnx \
--op_type Constant \
--opset ${OPSET} \
--op_name const_op_twenty_six \
--output_variables const_twenty_six_var int64 [1] \
--attributes value int64 [26] \
--output_onnx_file_path const_op_twenty_six.onnx \
--non_verbose

sog4onnx \
--op_type Constant \
--opset ${OPSET} \
--op_name const_op_thirty_five \
--output_variables const_thirty_five_var int64 [1] \
--attributes value int64 [35] \
--output_onnx_file_path const_op_thirty_five.onnx \
--non_verbose

sog4onnx \
--op_type Constant \
--opset ${OPSET} \
--op_name const_op_thirty_one \
--output_variables const_thirty_one_var int64 [1] \
--attributes value int64 [31] \
--output_onnx_file_path const_op_thirty_one.onnx \
--non_verbose

sog4onnx \
--op_type Constant \
--opset ${OPSET} \
--op_name const_op_thirty_two \
--output_variables const_thirty_two_var int64 [1] \
--attributes value int64 [32] \
--output_onnx_file_path const_op_thirty_two.onnx \
--non_verbose

sog4onnx \
--op_type Constant \
--opset ${OPSET} \
--op_name const_op_thirty_three \
--output_variables const_thirty_three_var int64 [1] \
--attributes value int64 [33] \
--output_onnx_file_path const_op_thirty_three.onnx \
--non_verbose

sog4onnx \
--op_type Constant \
--opset ${OPSET} \
--op_name const_op_thirty_four \
--output_variables const_thirty_four_var int64 [1] \
--attributes value int64 [34] \
--output_onnx_file_path const_op_thirty_four.onnx \
--non_verbose

################################################################################ Phase.1
PHASE1_RESOLUTIONS=(
    "1 35 15 20"
    "1 35 30 40"
    "1 35 60 80"
)

for((i=0; i<${#PHASE1_RESOLUTIONS[@]}; i++))
do
    RESOLUTION=(`echo ${PHASE1_RESOLUTIONS[i]}`)
    N=${RESOLUTION[0]}
    C=${RESOLUTION[1]}
    H=${RESOLUTION[2]}
    W=${RESOLUTION[3]}

    ########## Left
    sog4onnx \
    --op_type Slice \
    --opset ${OPSET} \
    --op_name phase1_slice_1_in_${H}x${W} \
    --input_variables phase1_slice_1_in_${H}x${W}_data float32 [${N},${C},${H},${W}] \
    --input_variables starts int64 [1] \
    --input_variables ends int64 [1] \
    --input_variables axes int64 [1] \
    --input_variables steps int64 [1] \
    --output_variables phase1_slice_1_out_${H}x${W} float32 [${N},4,${H},${W}] \
    --output_onnx_file_path phase1_slice_1_in_${H}x${W}.onnx \
    --non_verbose

    # starts
    snc4onnx \
    --input_onnx_file_paths const_op_zero.onnx phase1_slice_1_in_${H}x${W}.onnx \
    --output_onnx_file_path phase1_slice_1_in_${H}x${W}.onnx \
    --srcop_destop const_zero_var starts \
    --non_verbose

    # ends
    snc4onnx \
    --input_onnx_file_paths const_op_four.onnx phase1_slice_1_in_${H}x${W}.onnx \
    --output_onnx_file_path phase1_slice_1_in_${H}x${W}.onnx \
    --srcop_destop const_four_var ends \
    --non_verbose

    # axes
    snc4onnx \
    --input_onnx_file_paths const_op_one_1.onnx phase1_slice_1_in_${H}x${W}.onnx \
    --output_onnx_file_path phase1_slice_1_in_${H}x${W}.onnx \
    --srcop_destop const_one_var_1 axes \
    --non_verbose

    # steps
    snc4onnx \
    --input_onnx_file_paths const_op_one_2.onnx phase1_slice_1_in_${H}x${W}.onnx \
    --output_onnx_file_path phase1_slice_1_in_${H}x${W}.onnx \
    --srcop_destop const_one_var_2 steps \
    --non_verbose

    ########## Right
    sog4onnx \
    --op_type Slice \
    --opset ${OPSET} \
    --op_name phase1_slice_2_in_${H}x${W} \
    --input_variables phase1_slice_2_in_${H}x${W}_data float32 [${N},${C},${H},${W}] \
    --input_variables starts int64 [1] \
    --input_variables ends int64 [1] \
    --input_variables axes int64 [1] \
    --input_variables steps int64 [1] \
    --output_variables phase1_slice_2_out_${H}x${W} float32 [${N},9,${H},${W}] \
    --output_onnx_file_path phase1_slice_2_in_${H}x${W}.onnx \
    --non_verbose

    # starts
    snc4onnx \
    --input_onnx_file_paths const_op_twenty_six.onnx phase1_slice_2_in_${H}x${W}.onnx \
    --output_onnx_file_path phase1_slice_2_in_${H}x${W}.onnx \
    --srcop_destop const_twenty_six_var starts \
    --non_verbose

    # ends
    snc4onnx \
    --input_onnx_file_paths const_op_thirty_five.onnx phase1_slice_2_in_${H}x${W}.onnx \
    --output_onnx_file_path phase1_slice_2_in_${H}x${W}.onnx \
    --srcop_destop const_thirty_five_var ends \
    --non_verbose

    # axes
    snc4onnx \
    --input_onnx_file_paths const_op_one_1.onnx phase1_slice_2_in_${H}x${W}.onnx \
    --output_onnx_file_path phase1_slice_2_in_${H}x${W}.onnx \
    --srcop_destop const_one_var_1 axes \
    --non_verbose

    # steps
    snc4onnx \
    --input_onnx_file_paths const_op_one_2.onnx phase1_slice_2_in_${H}x${W}.onnx \
    --output_onnx_file_path phase1_slice_2_in_${H}x${W}.onnx \
    --srcop_destop const_one_var_2 steps \
    --non_verbose

    ########## Concat
    sog4onnx \
    --op_type Concat \
    --opset ${OPSET} \
    --op_name phase1_concat_in_${H}x${W} \
    --input_variables phase1_concat_in_1_${H}x${W} float32 [${N},4,${H},${W}] \
    --input_variables phase1_concat_in_2_${H}x${W} float32 [${N},22,${H},${W}] \
    --input_variables phase1_concat_in_3_${H}x${W} float32 [${N},9,${H},${W}] \
    --output_variables phase1_concat_out_${H}x${W} float32 [${N},35,${H},${W}] \
    --attributes axis int64 1 \
    --output_onnx_file_path phase1_concat_out_${H}x${W}.onnx \
    --non_verbose

    ########## Left + Right + Concat
    snc4onnx \
    --input_onnx_file_paths phase1_slice_1_in_${H}x${W}.onnx phase1_concat_out_${H}x${W}.onnx \
    --output_onnx_file_path phase1_concat_out_${H}x${W}.onnx \
    --srcop_destop phase1_slice_1_out_${H}x${W} phase1_concat_in_1_${H}x${W} \
    --non_verbose

    snc4onnx \
    --input_onnx_file_paths phase1_slice_2_in_${H}x${W}.onnx phase1_concat_out_${H}x${W}.onnx \
    --output_onnx_file_path phase1_concat_out_${H}x${W}.onnx \
    --srcop_destop phase1_slice_2_out_${H}x${W} phase1_concat_in_3_${H}x${W} \
    --non_verbose
done

########## Merge.1
snc4onnx \
--input_onnx_file_paths yolox_s_object_pose_ti_lite_640x480_57p75.onnx phase1_concat_out_15x20.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx \
--srcop_destop 1246 phase1_slice_1_in_15x20_data 1252 phase1_concat_in_2_15x20 1246 phase1_slice_2_in_15x20_data \
--non_verbose

########## Merge.2
snc4onnx \
--input_onnx_file_paths yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx phase1_concat_out_30x40.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx \
--srcop_destop 1051 phase1_slice_1_in_30x40_data 1057 phase1_concat_in_2_30x40 1051 phase1_slice_2_in_30x40_data \
--non_verbose

########## Merge.3
snc4onnx \
--input_onnx_file_paths yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx phase1_concat_out_60x80.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx \
--srcop_destop 856 phase1_slice_1_in_60x80_data 862 phase1_concat_in_2_60x80 856 phase1_slice_2_in_60x80_data \
--non_verbose

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx

########## Merge.4
for((i=0; i<5; i++))
do
    svs4onnx \
    --input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx \
    --from_output_variable_name phase1_concat_out_15x20 \
    --to_input_variable_name 1312 \
    --output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx \
    --non_verbose
done

for((i=0; i<5; i++))
do
    svs4onnx \
    --input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx \
    --from_output_variable_name phase1_concat_out_30x40 \
    --to_input_variable_name 1117 \
    --output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx \
    --non_verbose
done

for((i=0; i<5; i++))
do
    svs4onnx \
    --input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx \
    --from_output_variable_name phase1_concat_out_60x80 \
    --to_input_variable_name 922 \
    --output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx \
    --non_verbose
done

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx






################################################################################ Phase.2
PHASE2_RESOLUTIONS=(
    "1 35 15 20"
    "1 35 30 40"
    "1 35 60 80"
)

for((i=0; i<${#PHASE2_RESOLUTIONS[@]}; i++))
do
    RESOLUTION=(`echo ${PHASE2_RESOLUTIONS[i]}`)
    N=${RESOLUTION[0]}
    C=${RESOLUTION[1]}
    H=${RESOLUTION[2]}
    W=${RESOLUTION[3]}

    ########## Left
    sog4onnx \
    --op_type Slice \
    --opset ${OPSET} \
    --op_name phase2_slice_1_in_${H}x${W} \
    --input_variables phase2_slice_1_in_${H}x${W}_data float32 [${N},${C},${H},${W}] \
    --input_variables starts int64 [1] \
    --input_variables ends int64 [1] \
    --input_variables axes int64 [1] \
    --input_variables steps int64 [1] \
    --output_variables phase2_slice_1_out_${H}x${W} float32 [${N},25,${H},${W}] \
    --output_onnx_file_path phase2_slice_1_in_${H}x${W}.onnx \
    --non_verbose

    # starts
    snc4onnx \
    --input_onnx_file_paths const_op_zero.onnx phase2_slice_1_in_${H}x${W}.onnx \
    --output_onnx_file_path phase2_slice_1_in_${H}x${W}.onnx \
    --srcop_destop const_zero_var starts \
    --non_verbose

    # ends
    snc4onnx \
    --input_onnx_file_paths const_op_twenty_five.onnx phase2_slice_1_in_${H}x${W}.onnx \
    --output_onnx_file_path phase2_slice_1_in_${H}x${W}.onnx \
    --srcop_destop const_twenty_five_var ends \
    --non_verbose

    # axes
    snc4onnx \
    --input_onnx_file_paths const_op_one_1.onnx phase2_slice_1_in_${H}x${W}.onnx \
    --output_onnx_file_path phase2_slice_1_in_${H}x${W}.onnx \
    --srcop_destop const_one_var_1 axes \
    --non_verbose

    # steps
    snc4onnx \
    --input_onnx_file_paths const_op_one_2.onnx phase2_slice_1_in_${H}x${W}.onnx \
    --output_onnx_file_path phase2_slice_1_in_${H}x${W}.onnx \
    --srcop_destop const_one_var_2 steps \
    --non_verbose

    ########## Right
    sog4onnx \
    --op_type Slice \
    --opset ${OPSET} \
    --op_name phase2_slice_2_in_${H}x${W} \
    --input_variables phase2_slice_2_in_${H}x${W}_data float32 [${N},${C},${H},${W}] \
    --input_variables starts int64 [1] \
    --input_variables ends int64 [1] \
    --input_variables axes int64 [1] \
    --input_variables steps int64 [1] \
    --output_variables phase2_slice_2_out_${H}x${W} float32 [${N},4,${H},${W}] \
    --output_onnx_file_path phase2_slice_2_in_${H}x${W}.onnx \
    --non_verbose

    # starts
    snc4onnx \
    --input_onnx_file_paths const_op_thirty_one.onnx phase2_slice_2_in_${H}x${W}.onnx \
    --output_onnx_file_path phase2_slice_2_in_${H}x${W}.onnx \
    --srcop_destop const_thirty_one_var starts \
    --non_verbose

    # ends
    snc4onnx \
    --input_onnx_file_paths const_op_thirty_five.onnx phase2_slice_2_in_${H}x${W}.onnx \
    --output_onnx_file_path phase2_slice_2_in_${H}x${W}.onnx \
    --srcop_destop const_thirty_five_var ends \
    --non_verbose

    # axes
    snc4onnx \
    --input_onnx_file_paths const_op_one_1.onnx phase2_slice_2_in_${H}x${W}.onnx \
    --output_onnx_file_path phase2_slice_2_in_${H}x${W}.onnx \
    --srcop_destop const_one_var_1 axes \
    --non_verbose

    # steps
    snc4onnx \
    --input_onnx_file_paths const_op_one_2.onnx phase2_slice_2_in_${H}x${W}.onnx \
    --output_onnx_file_path phase2_slice_2_in_${H}x${W}.onnx \
    --srcop_destop const_one_var_2 steps \
    --non_verbose


    ########## Concat
    sog4onnx \
    --op_type Concat \
    --opset ${OPSET} \
    --op_name phase2_concat_in_${H}x${W} \
    --input_variables phase2_concat_in_1_${H}x${W} float32 [${N},25,${H},${W}] \
    --input_variables phase2_concat_in_2_${H}x${W} float32 [${N},6,${H},${W}] \
    --input_variables phase2_concat_in_3_${H}x${W} float32 [${N},4,${H},${W}] \
    --output_variables phase2_concat_out_${H}x${W} float32 [${N},35,${H},${W}] \
    --attributes axis int64 1 \
    --output_onnx_file_path phase2_concat_out_${H}x${W}.onnx \
    --non_verbose

    ########## Left + Right + Concat
    snc4onnx \
    --input_onnx_file_paths phase2_slice_1_in_${H}x${W}.onnx phase2_concat_out_${H}x${W}.onnx \
    --output_onnx_file_path phase2_concat_out_${H}x${W}.onnx \
    --srcop_destop phase2_slice_1_out_${H}x${W} phase2_concat_in_1_${H}x${W} \
    --non_verbose

    snc4onnx \
    --input_onnx_file_paths phase2_slice_2_in_${H}x${W}.onnx phase2_concat_out_${H}x${W}.onnx \
    --output_onnx_file_path phase2_concat_out_${H}x${W}.onnx \
    --srcop_destop phase2_slice_2_out_${H}x${W} phase2_concat_in_3_${H}x${W} \
    --non_verbose
done

########## Merge.1
snc4onnx \
--input_onnx_file_paths yolox_s_object_pose_ti_lite_640x480_57p75_concat1.onnx phase2_concat_out_15x20.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--srcop_destop 1341 phase2_concat_in_2_15x20 phase1_concat_out_15x20 phase2_slice_1_in_15x20_data phase1_concat_out_15x20 phase2_concat_in_2_15x20 \
--op_prefixes_after_merging phase1_ phase2_ \
--non_verbose

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx

for((i=0; i<5; i++))
do
    sor4onnx \
    --input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
    --old_new "phase1__" "" \
    --mode full \
    --search_mode prefix_match \
    --output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
    --non_verbose

    sor4onnx \
    --input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
    --old_new "phase2__" "" \
    --mode full \
    --search_mode prefix_match \
    --output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
    --non_verbose
done

svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--from_output_variable_name phase1_concat_out_15x20 \
--to_input_variable_name phase2_slice_2_in_15x20_data \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--non_verbose
onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx

svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--from_output_variable_name phase2_concat_out_15x20 \
--to_input_variable_name 1401 \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--non_verbose
onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx

snd4onnx \
--remove_node_names phase2_slice_2_in_15x20_data \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--non_verbose

########## Merge.2
snc4onnx \
--input_onnx_file_paths yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx phase2_concat_out_30x40.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--srcop_destop 1146 phase2_concat_in_2_30x40 phase1_concat_out_30x40 phase2_slice_1_in_30x40_data phase1_concat_out_30x40 phase2_concat_in_2_30x40 \
--op_prefixes_after_merging phase1_ phase2_ \
--non_verbose

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx

for((i=0; i<5; i++))
do
    sor4onnx \
    --input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
    --old_new "phase1__" "" \
    --mode full \
    --search_mode prefix_match \
    --output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
    --non_verbose

    sor4onnx \
    --input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
    --old_new "phase2__" "" \
    --mode full \
    --search_mode prefix_match \
    --output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
    --non_verbose
done

svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--from_output_variable_name phase1_concat_out_30x40 \
--to_input_variable_name phase2_slice_2_in_30x40_data \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--non_verbose
onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx

svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--from_output_variable_name phase2_concat_out_30x40 \
--to_input_variable_name 1206 \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--non_verbose
onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx

snd4onnx \
--remove_node_names phase2_slice_2_in_30x40_data \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--non_verbose

########## Merge.3
snc4onnx \
--input_onnx_file_paths yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx phase2_concat_out_60x80.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--srcop_destop 951 phase2_concat_in_2_60x80 phase1_concat_out_60x80 phase2_slice_1_in_60x80_data phase1_concat_out_60x80 phase2_concat_in_2_60x80 \
--op_prefixes_after_merging phase1_ phase2_ \
--non_verbose

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx

for((i=0; i<5; i++))
do
    sor4onnx \
    --input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
    --old_new "phase1__" "" \
    --mode full \
    --search_mode prefix_match \
    --output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
    --non_verbose

    sor4onnx \
    --input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
    --old_new "phase2__" "" \
    --mode full \
    --search_mode prefix_match \
    --output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
    --non_verbose
done

svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--from_output_variable_name phase1_concat_out_60x80 \
--to_input_variable_name phase2_slice_2_in_60x80_data \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--non_verbose
onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx

svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--from_output_variable_name phase2_concat_out_60x80 \
--to_input_variable_name 1011 \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--non_verbose
onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx

snd4onnx \
--remove_node_names phase2_slice_2_in_60x80_data \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--non_verbose





################################################################################ Phase.3
svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_concat2.onnx \
--from_output_variable_name 1427 \
--to_input_variable_name 1552 \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon.onnx \
--non_verbose

svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon.onnx \
--from_output_variable_name 1427 \
--to_input_variable_name 1609 \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon.onnx \
--non_verbose

svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon.onnx \
--from_output_variable_name 1427 \
--to_input_variable_name 1688 \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon.onnx \
--non_verbose

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_recon.onnx yolox_s_object_pose_ti_lite_640x480_57p75_recon.onnx




sog4onnx \
--op_type Slice \
--opset ${OPSET} \
--op_name phase3_slice_in_1x6300x35 \
--input_variables phase3_slice_in_1x6300x35_data float32 [1,6300,35] \
--input_variables starts int64 [1] \
--input_variables ends int64 [1] \
--input_variables axes int64 [1] \
--input_variables steps int64 [1] \
--output_variables phase3_slice_out_1x6300x35 float32 [1,6300,28] \
--output_onnx_file_path phase3_slice_in_1x6300x35.onnx \
--non_verbose

# starts
snc4onnx \
--input_onnx_file_paths const_op_four.onnx phase3_slice_in_1x6300x35.onnx \
--output_onnx_file_path phase3_slice_in_1x6300x35.onnx \
--srcop_destop const_four_var starts \
--non_verbose

# ends
snc4onnx \
--input_onnx_file_paths const_op_thirty_two.onnx phase3_slice_in_1x6300x35.onnx \
--output_onnx_file_path phase3_slice_in_1x6300x35.onnx \
--srcop_destop const_thirty_two_var ends \
--non_verbose

# axes
snc4onnx \
--input_onnx_file_paths const_op_two.onnx phase3_slice_in_1x6300x35.onnx \
--output_onnx_file_path phase3_slice_in_1x6300x35.onnx \
--srcop_destop const_two_var axes \
--non_verbose

# steps
snc4onnx \
--input_onnx_file_paths const_op_one_1.onnx phase3_slice_in_1x6300x35.onnx \
--output_onnx_file_path phase3_slice_in_1x6300x35.onnx \
--srcop_destop const_one_var_1 steps \
--non_verbose

onnxsim phase3_slice_in_1x6300x35.onnx phase3_slice_in_1x6300x35.onnx

########## Concat
sog4onnx \
--op_type Concat \
--opset ${OPSET} \
--op_name phase3_concat_in_1x6300x35 \
--input_variables phase3_concat_in_1_1x6300x35 float32 [1,6300,2] \
--input_variables phase3_concat_in_2_1x6300x35 float32 [1,6300,2] \
--input_variables phase3_concat_in_3_1x6300x35 float32 [1,6300,28] \
--input_variables phase3_concat_in_4_1x6300x35 float32 [1,6300,2] \
--input_variables phase3_concat_in_5_1x6300x35 float32 [1,6300,1] \
--output_variables phase3_concat_out_1x6300x35 float32 [1,6300,35] \
--attributes axis int64 2 \
--output_onnx_file_path phase3_concat_out_1x6300x35.onnx \
--non_verbose

########## Merge
snc4onnx \
--input_onnx_file_paths phase3_slice_in_1x6300x35.onnx phase3_concat_out_1x6300x35.onnx \
--output_onnx_file_path phase3_concat_out_1x6300x35.onnx \
--srcop_destop phase3_slice_out_1x6300x35 phase3_concat_in_3_1x6300x35 \
--non_verbose

snc4onnx \
--input_onnx_file_paths yolox_s_object_pose_ti_lite_640x480_57p75_recon.onnx phase3_concat_out_1x6300x35.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc.onnx \
--srcop_destop 1427 phase3_slice_in_1x6300x35_data 1503 phase3_concat_in_1_1x6300x35 1696 phase3_concat_in_2_1x6300x35 1560 phase3_concat_in_4_1x6300x35 1615 phase3_concat_in_5_1x6300x35 \
--non_verbose

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc.onnx yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc.onnx


for((i=0; i<5; i++))
do
    svs4onnx \
    --input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc.onnx \
    --from_output_variable_name phase3_concat_out_1x6300x35 \
    --to_input_variable_name 1769 \
    --output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc.onnx \
    --non_verbose
done

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc.onnx yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc.onnx



################################################################################ Phase.4
sog4onnx \
--op_type Slice \
--opset ${OPSET} \
--op_name phase4_slice_in_1x6300x35 \
--input_variables phase4_slice_in_1x6300x35_data float32 [1,6300,35] \
--input_variables starts int64 [1] \
--input_variables ends int64 [1] \
--input_variables axes int64 [1] \
--input_variables steps int64 [1] \
--output_variables phase4_slice_out_1x6300x35 float32 [1,6300,31] \
--output_onnx_file_path phase4_slice_in_1x6300x35.onnx \
--non_verbose

# starts
snc4onnx \
--input_onnx_file_paths const_op_four.onnx phase4_slice_in_1x6300x35.onnx \
--output_onnx_file_path phase4_slice_in_1x6300x35.onnx \
--srcop_destop const_four_var starts \
--non_verbose

# ends
snc4onnx \
--input_onnx_file_paths const_op_thirty_five.onnx phase4_slice_in_1x6300x35.onnx \
--output_onnx_file_path phase4_slice_in_1x6300x35.onnx \
--srcop_destop const_thirty_five_var ends \
--non_verbose

# axes
snc4onnx \
--input_onnx_file_paths const_op_two.onnx phase4_slice_in_1x6300x35.onnx \
--output_onnx_file_path phase4_slice_in_1x6300x35.onnx \
--srcop_destop const_two_var axes \
--non_verbose

# steps
snc4onnx \
--input_onnx_file_paths const_op_one_1.onnx phase4_slice_in_1x6300x35.onnx \
--output_onnx_file_path phase4_slice_in_1x6300x35.onnx \
--srcop_destop const_one_var_1 steps \
--non_verbose

onnxsim phase4_slice_in_1x6300x35.onnx phase4_slice_in_1x6300x35.onnx


########## Concat
sog4onnx \
--op_type Concat \
--opset ${OPSET} \
--op_name phase4_concat_in_1x6300x35 \
--input_variables phase4_concat_in_1_1x6300x35 float32 [1,6300,4] \
--input_variables phase4_concat_in_2_1x6300x35 float32 [1,6300,31] \
--output_variables phase4_concat_out_1x6300x35 float32 [1,6300,35] \
--attributes axis int64 2 \
--output_onnx_file_path phase4_concat_out_1x6300x35.onnx \
--non_verbose

########## Merge
snc4onnx \
--input_onnx_file_paths phase4_slice_in_1x6300x35.onnx phase4_concat_out_1x6300x35.onnx \
--output_onnx_file_path phase4_concat_out_1x6300x35.onnx \
--srcop_destop phase4_slice_out_1x6300x35 phase4_concat_in_2_1x6300x35 \
--non_verbose

snc4onnx \
--input_onnx_file_paths yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc.onnx phase4_concat_out_1x6300x35.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc2.onnx \
--srcop_destop 1800 phase4_concat_in_1_1x6300x35 phase3_concat_out_1x6300x35 phase4_slice_in_1x6300x35_data \
--non_verbose

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc2.onnx yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc2.onnx


for((i=0; i<4; i++))
do
    svs4onnx \
    --input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc2.onnx \
    --from_output_variable_name phase4_concat_out_1x6300x35 \
    --to_input_variable_name 1873 \
    --output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc2.onnx \
    --non_verbose
done

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc2.onnx yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc2.onnx



################################################################################ Phase.5
########## Left
sog4onnx \
--op_type Slice \
--opset ${OPSET} \
--op_name phase5_slice_1_in_1x6300x35 \
--input_variables phase5_slice_1_in_1x6300x35_data float32 [1,6300,35] \
--input_variables starts int64 [1] \
--input_variables ends int64 [1] \
--input_variables axes int64 [1] \
--input_variables steps int64 [1] \
--output_variables phase5_slice_1_out_1x6300x35 float32 [1,6300,32] \
--output_onnx_file_path phase5_slice_1_in_1x6300x35.onnx \
--non_verbose

# starts
snc4onnx \
--input_onnx_file_paths const_op_zero.onnx phase5_slice_1_in_1x6300x35.onnx \
--output_onnx_file_path phase5_slice_1_in_1x6300x35.onnx \
--srcop_destop const_zero_var starts \
--non_verbose

# ends
snc4onnx \
--input_onnx_file_paths const_op_thirty_two.onnx phase5_slice_1_in_1x6300x35.onnx \
--output_onnx_file_path phase5_slice_1_in_1x6300x35.onnx \
--srcop_destop const_thirty_two_var ends \
--non_verbose

# axes
snc4onnx \
--input_onnx_file_paths const_op_two.onnx phase5_slice_1_in_1x6300x35.onnx \
--output_onnx_file_path phase5_slice_1_in_1x6300x35.onnx \
--srcop_destop const_two_var axes \
--non_verbose

# steps
snc4onnx \
--input_onnx_file_paths const_op_one_2.onnx phase5_slice_1_in_1x6300x35.onnx \
--output_onnx_file_path phase5_slice_1_in_1x6300x35.onnx \
--srcop_destop const_one_var_2 steps \
--non_verbose

onnxsim phase5_slice_1_in_1x6300x35.onnx phase5_slice_1_in_1x6300x35.onnx

########## Right
sog4onnx \
--op_type Slice \
--opset ${OPSET} \
--op_name phase5_slice_2_in_1x6300x35 \
--input_variables phase5_slice_2_in_1x6300x35_data float32 [1,6300,35] \
--input_variables starts int64 [1] \
--input_variables ends int64 [1] \
--input_variables axes int64 [1] \
--input_variables steps int64 [1] \
--output_variables phase5_slice_2_out_1x6300x35 float32 [1,6300,2] \
--output_onnx_file_path phase5_slice_2_in_1x6300x35.onnx \
--non_verbose

# starts
snc4onnx \
--input_onnx_file_paths const_op_thirty_three.onnx phase5_slice_2_in_1x6300x35.onnx \
--output_onnx_file_path phase5_slice_2_in_1x6300x35.onnx \
--srcop_destop const_thirty_three_var starts \
--non_verbose

# ends
snc4onnx \
--input_onnx_file_paths const_op_thirty_five.onnx phase5_slice_2_in_1x6300x35.onnx \
--output_onnx_file_path phase5_slice_2_in_1x6300x35.onnx \
--srcop_destop const_thirty_five_var ends \
--non_verbose

# axes
snc4onnx \
--input_onnx_file_paths const_op_two.onnx phase5_slice_2_in_1x6300x35.onnx \
--output_onnx_file_path phase5_slice_2_in_1x6300x35.onnx \
--srcop_destop const_two_var axes \
--non_verbose

# steps
snc4onnx \
--input_onnx_file_paths const_op_one_2.onnx phase5_slice_2_in_1x6300x35.onnx \
--output_onnx_file_path phase5_slice_2_in_1x6300x35.onnx \
--srcop_destop const_one_var_2 steps \
--non_verbose

onnxsim phase5_slice_2_in_1x6300x35.onnx phase5_slice_2_in_1x6300x35.onnx


########## Concat
sog4onnx \
--op_type Concat \
--opset ${OPSET} \
--op_name phase5_concat_in_1x6300x35 \
--input_variables phase5_concat_in_1_1x6300x35 float32 [1,6300,32] \
--input_variables phase5_concat_in_2_1x6300x35 float32 [1,6300,1] \
--input_variables phase5_concat_in_3_1x6300x35 float32 [1,6300,2] \
--output_variables phase5_concat_out_1x6300x35 float32 [1,6300,35] \
--attributes axis int64 2 \
--output_onnx_file_path phase5_concat_out_1x6300x35.onnx \
--non_verbose

########## Merge
snc4onnx \
--input_onnx_file_paths phase5_slice_1_in_1x6300x35.onnx phase5_concat_out_1x6300x35.onnx \
--output_onnx_file_path phase5_concat_out_1x6300x35.onnx \
--srcop_destop phase5_slice_1_out_1x6300x35 phase5_concat_in_1_1x6300x35 \
--non_verbose

snc4onnx \
--input_onnx_file_paths phase5_slice_2_in_1x6300x35.onnx phase5_concat_out_1x6300x35.onnx \
--output_onnx_file_path phase5_concat_out_1x6300x35.onnx \
--srcop_destop phase5_slice_2_out_1x6300x35 phase5_concat_in_3_1x6300x35 \
--non_verbose



snc4onnx \
--input_onnx_file_paths yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc2.onnx phase5_concat_out_1x6300x35.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc3.onnx \
--srcop_destop phase4_concat_out_1x6300x35 phase5_slice_1_in_1x6300x35_data  \
--non_verbose

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc3.onnx yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc3.onnx

###### OK

svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc3.onnx \
--from_output_variable_name 1950 \
--to_input_variable_name phase5_concat_in_2_1x6300x35 \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc3.onnx \
--non_verbose

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc3.onnx yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc3.onnx

###### OK

snd4onnx \
--remove_node_names phase5_concat_in_2_1x6300x35 \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc3.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc3.onnx \
--non_verbose

###### OK

svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc3.onnx \
--from_output_variable_name phase5_concat_out_1x6300x35 \
--to_input_variable_name 1951 \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc3.onnx \
--non_verbose

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc3.onnx yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc3.onnx

###### OK

################################################################################ Phase.6
########## Left
sog4onnx \
--op_type Slice \
--opset ${OPSET} \
--op_name phase6_slice_1_in_1x6300x35 \
--input_variables phase6_slice_1_in_1x6300x35_data float32 [1,6300,35] \
--input_variables starts int64 [1] \
--input_variables ends int64 [1] \
--input_variables axes int64 [1] \
--input_variables steps int64 [1] \
--output_variables phase6_slice_1_out_1x6300x35 float32 [1,6300,33] \
--output_onnx_file_path phase6_slice_1_in_1x6300x35.onnx \
--non_verbose

# starts
snc4onnx \
--input_onnx_file_paths const_op_zero.onnx phase6_slice_1_in_1x6300x35.onnx \
--output_onnx_file_path phase6_slice_1_in_1x6300x35.onnx \
--srcop_destop const_zero_var starts \
--non_verbose

# ends
snc4onnx \
--input_onnx_file_paths const_op_thirty_three.onnx phase6_slice_1_in_1x6300x35.onnx \
--output_onnx_file_path phase6_slice_1_in_1x6300x35.onnx \
--srcop_destop const_thirty_three_var ends \
--non_verbose

# axes
snc4onnx \
--input_onnx_file_paths const_op_two.onnx phase6_slice_1_in_1x6300x35.onnx \
--output_onnx_file_path phase6_slice_1_in_1x6300x35.onnx \
--srcop_destop const_two_var axes \
--non_verbose

# steps
snc4onnx \
--input_onnx_file_paths const_op_one_2.onnx phase6_slice_1_in_1x6300x35.onnx \
--output_onnx_file_path phase6_slice_1_in_1x6300x35.onnx \
--srcop_destop const_one_var_2 steps \
--non_verbose

onnxsim phase6_slice_1_in_1x6300x35.onnx phase6_slice_1_in_1x6300x35.onnx

########## Right
sog4onnx \
--op_type Slice \
--opset ${OPSET} \
--op_name phase6_slice_2_in_1x6300x35 \
--input_variables phase6_slice_2_in_1x6300x35_data float32 [1,6300,35] \
--input_variables starts int64 [1] \
--input_variables ends int64 [1] \
--input_variables axes int64 [1] \
--input_variables steps int64 [1] \
--output_variables phase6_slice_2_out_1x6300x35 float32 [1,6300,1] \
--output_onnx_file_path phase6_slice_2_in_1x6300x35.onnx \
--non_verbose

# starts
snc4onnx \
--input_onnx_file_paths const_op_thirty_four.onnx phase6_slice_2_in_1x6300x35.onnx \
--output_onnx_file_path phase6_slice_2_in_1x6300x35.onnx \
--srcop_destop const_thirty_four_var starts \
--non_verbose

# ends
snc4onnx \
--input_onnx_file_paths const_op_thirty_five.onnx phase6_slice_2_in_1x6300x35.onnx \
--output_onnx_file_path phase6_slice_2_in_1x6300x35.onnx \
--srcop_destop const_thirty_five_var ends \
--non_verbose

# axes
snc4onnx \
--input_onnx_file_paths const_op_two.onnx phase6_slice_2_in_1x6300x35.onnx \
--output_onnx_file_path phase6_slice_2_in_1x6300x35.onnx \
--srcop_destop const_two_var axes \
--non_verbose

# steps
snc4onnx \
--input_onnx_file_paths const_op_one_2.onnx phase6_slice_2_in_1x6300x35.onnx \
--output_onnx_file_path phase6_slice_2_in_1x6300x35.onnx \
--srcop_destop const_one_var_2 steps \
--non_verbose

onnxsim phase6_slice_2_in_1x6300x35.onnx phase6_slice_2_in_1x6300x35.onnx

########## Concat
sog4onnx \
--op_type Concat \
--opset ${OPSET} \
--op_name phase6_concat_in_1x6300x35 \
--input_variables phase6_concat_in_1_1x6300x35 float32 [1,6300,33] \
--input_variables phase6_concat_in_2_1x6300x35 float32 [1,6300,1] \
--input_variables phase6_concat_in_3_1x6300x35 float32 [1,6300,1] \
--output_variables phase6_concat_out_1x6300x35 float32 [1,6300,35] \
--attributes axis int64 2 \
--output_onnx_file_path phase6_concat_out_1x6300x35.onnx \
--non_verbose

########## Merge
snc4onnx \
--input_onnx_file_paths phase6_slice_1_in_1x6300x35.onnx phase6_concat_out_1x6300x35.onnx \
--output_onnx_file_path phase6_concat_out_1x6300x35.onnx \
--srcop_destop phase6_slice_1_out_1x6300x35 phase6_concat_in_1_1x6300x35 \
--non_verbose

snc4onnx \
--input_onnx_file_paths phase6_slice_2_in_1x6300x35.onnx phase6_concat_out_1x6300x35.onnx \
--output_onnx_file_path phase6_concat_out_1x6300x35.onnx \
--srcop_destop phase6_slice_2_out_1x6300x35 phase6_concat_in_2_1x6300x35 \
--non_verbose

onnxsim phase6_concat_out_1x6300x35.onnx phase6_concat_out_1x6300x35.onnx

snc4onnx \
--input_onnx_file_paths yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc3.onnx phase6_concat_out_1x6300x35.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc4.onnx \
--srcop_destop 2010 phase6_concat_in_3_1x6300x35 phase5_concat_out_1x6300x35 phase6_slice_1_in_1x6300x35_data phase5_concat_out_1x6300x35 phase6_slice_2_in_1x6300x35_data \
--non_verbose

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc4.onnx yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc4.onnx


svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc4.onnx \
--from_output_variable_name phase6_concat_out_1x6300x35 \
--to_input_variable_name 2011 \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc4.onnx \
--non_verbose

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc4.onnx yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc4.onnx

###### OK


################################################################################ Phase.7
########## Left
sog4onnx \
--op_type Slice \
--opset ${OPSET} \
--op_name phase7_slice_1_in_1x6300x35 \
--input_variables phase7_slice_1_in_1x6300x35_data float32 [1,6300,35] \
--input_variables starts int64 [1] \
--input_variables ends int64 [1] \
--input_variables axes int64 [1] \
--input_variables steps int64 [1] \
--output_variables phase7_slice_1_out_1x6300x35 float32 [1,6300,34] \
--output_onnx_file_path phase7_slice_1_in_1x6300x35.onnx \
--non_verbose

# starts
snc4onnx \
--input_onnx_file_paths const_op_zero.onnx phase7_slice_1_in_1x6300x35.onnx \
--output_onnx_file_path phase7_slice_1_in_1x6300x35.onnx \
--srcop_destop const_zero_var starts \
--non_verbose

# ends
snc4onnx \
--input_onnx_file_paths const_op_thirty_four.onnx phase7_slice_1_in_1x6300x35.onnx \
--output_onnx_file_path phase7_slice_1_in_1x6300x35.onnx \
--srcop_destop const_thirty_four_var ends \
--non_verbose

# axes
snc4onnx \
--input_onnx_file_paths const_op_two.onnx phase7_slice_1_in_1x6300x35.onnx \
--output_onnx_file_path phase7_slice_1_in_1x6300x35.onnx \
--srcop_destop const_two_var axes \
--non_verbose

# steps
snc4onnx \
--input_onnx_file_paths const_op_one_2.onnx phase7_slice_1_in_1x6300x35.onnx \
--output_onnx_file_path phase7_slice_1_in_1x6300x35.onnx \
--srcop_destop const_one_var_2 steps \
--non_verbose

onnxsim phase7_slice_1_in_1x6300x35.onnx phase7_slice_1_in_1x6300x35.onnx


########## Concat
sog4onnx \
--op_type Concat \
--opset ${OPSET} \
--op_name phase7_concat_in_1x6300x35 \
--input_variables phase7_concat_in_1_1x6300x35 float32 [1,6300,34] \
--input_variables phase7_concat_in_2_1x6300x35 float32 [1,6300,1] \
--output_variables phase7_concat_out_1x6300x35 float32 [1,6300,35] \
--attributes axis int64 2 \
--output_onnx_file_path phase7_concat_out_1x6300x35.onnx \
--non_verbose

onnxsim phase7_concat_out_1x6300x35.onnx phase7_concat_out_1x6300x35.onnx

########## Merge
snc4onnx \
--input_onnx_file_paths phase7_slice_1_in_1x6300x35.onnx phase7_concat_out_1x6300x35.onnx \
--output_onnx_file_path phase7_concat_out_1x6300x35.onnx \
--srcop_destop phase7_slice_1_out_1x6300x35 phase7_concat_in_1_1x6300x35 \
--non_verbose

onnxsim phase7_concat_out_1x6300x35.onnx phase7_concat_out_1x6300x35.onnx

snc4onnx \
--input_onnx_file_paths yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc4.onnx phase7_concat_out_1x6300x35.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc5.onnx \
--srcop_destop 2070 phase7_concat_in_2_1x6300x35 phase6_concat_out_1x6300x35 phase7_slice_1_in_1x6300x35_data \
--non_verbose

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc5.onnx yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc5.onnx

svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc5.onnx \
--from_output_variable_name phase7_concat_out_1x6300x35 \
--to_input_variable_name 2071 \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc5.onnx \
--non_verbose

onnxsim yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc5.onnx yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc5.onnx

##### OK

cp yolox_s_object_pose_ti_lite_640x480_57p75_recon_conc5.onnx yolox_s_object_pose_ti_lite_640x480_57p75_opt.onnx

####################################

sam4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_opt.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_opt.onnx \
--op_name NonMaxSuppression_1294 \
--input_constants 918 int64 [20]





############################################
svs4onnx \
--input_onnx_file_path _yolox_s_object_pose_ti_lite_640x480_57p75_opt_float32.onnx \
--from_output_variable_name model/tf.concat_24/concat \
--to_input_variable_name inputs_1 \
--output_onnx_file_path _yolox_s_object_pose_ti_lite_640x480_57p75_dynamic.onnx \
--non_verbose

onnx2json \
--input_onnx_file_path _yolox_s_object_pose_ti_lite_640x480_57p75_opt_float32.onnx \
--output_json_path _yolox_s_object_pose_ti_lite_640x480_57p75_opt_float32.json \
--json_indent 2


################################################
svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_opt.onnx \
--from_output_variable_name 2103 \
--to_input_variable_name 2106 \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_opt_.onnx \
--non_verbose

svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_opt_.onnx \
--from_output_variable_name 2103 \
--to_input_variable_name 2106 \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_opt_.onnx \
--non_verbose

svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_opt_.onnx \
--from_output_variable_name 2103 \
--to_input_variable_name 2106 \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_opt_.onnx \
--non_verbose

svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_opt_.onnx \
--from_output_variable_name 2103 \
--to_input_variable_name 2106 \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_opt_.onnx \
--non_verbose

sam4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_opt_.onnx \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_opt_.onnx \
--op_name NonMaxSuppression_1294 \
--input_constants 2126 float32 [0.4]


onnx2json \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_pinto_dynamic_ver2.onnx \
--output_json_path yolox_s_object_pose_ti_lite_640x480_57p75_pinto_dynamic_ver2.json \
--json_indent 2

json2onnx \
--input_json_path yolox_s_object_pose_ti_lite_640x480_57p75_pinto_dynamic_ver2.json \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_pinto_dynamic_ver2.onnx


################################################
svs4onnx \
--input_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_opt.onnx \
--from_output_variable_name phase3_concat_out_1x6300x35 \
--to_input_variable_name phase5_slice_2_in_1x6300x35_data \
--output_onnx_file_path yolox_s_object_pose_ti_lite_640x480_57p75_pinto_dynamic_ver3.onnx \
--non_verbose



flatc \
-t \
--strict-json \
--defaults-json \
-o . \
schema.fbs -- yolox_s_object_pose_ti_lite_640x480_57p75_opt__float32.tflite

flatc \
-o . \
-b schema.fbs yolox_s_object_pose_ti_lite_640x480_57p75_opt__float32.json


python -m tf2onnx.convert \
--opset 11 \
--inputs-as-nchw inputs_0 \
--tflite yolox_s_object_pose_ti_lite_640x480_57p75_opt_float32.tflite \
--output yolox_s_object_pose_ti_lite_640x480_57p75_opt_float32.onnx


