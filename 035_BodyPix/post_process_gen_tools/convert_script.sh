#!/bin/bash

# pip install tensorflowjs -U --no-deps
# pip install tf2onnx -U --no-deps
# pip install onnxsim==0.4.33

onnx_export() {
    tfjs_graph_converter \
    ${MODEL_TYPE}/model.json \
    saved_model_${MODEL_TYPE} \
    --output_format tf_saved_model \
    --compat_mode tflite

    python -m tf2onnx.convert \
    --saved-model saved_model_${MODEL_TYPE} \
    --opset 11 \
    --inputs-as-nchw sub_2 \
    --output ${FILE_NAME}.onnx

    onnxsim ${FILE_NAME}.onnx ${FILE_NAME}.onnx

    sor4onnx \
    --input_onnx_file_path ${FILE_NAME}.onnx \
    --old_new "sub_2" "input" \
    --mode inputs \
    --search_mode prefix_match \
    --output_onnx_file_path ${FILE_NAME}.onnx
    sor4onnx \
    --input_onnx_file_path ${FILE_NAME}.onnx \
    --old_new "float_" "" \
    --mode outputs \
    --search_mode prefix_match \
    --output_onnx_file_path ${FILE_NAME}.onnx
    sor4onnx \
    --input_onnx_file_path ${FILE_NAME}.onnx \
    --old_new "resnet_v1_50/displacement_bwd_2/BiasAdd" "displacement_bwd" \
    --mode outputs \
    --search_mode prefix_match \
    --output_onnx_file_path ${FILE_NAME}.onnx
    sor4onnx \
    --input_onnx_file_path ${FILE_NAME}.onnx \
    --old_new "resnet_v1_50/displacement_fwd_2/BiasAdd" "displacement_fwd" \
    --mode outputs \
    --search_mode prefix_match \
    --output_onnx_file_path ${FILE_NAME}.onnx
    sor4onnx \
    --input_onnx_file_path ${FILE_NAME}.onnx \
    --old_new "MobilenetV1/displacement_bwd_2/BiasAdd" "displacement_bwd" \
    --mode outputs \
    --search_mode prefix_match \
    --output_onnx_file_path ${FILE_NAME}.onnx
    sor4onnx \
    --input_onnx_file_path ${FILE_NAME}.onnx \
    --old_new "MobilenetV1/displacement_fwd_2/BiasAdd" "displacement_fwd" \
    --mode outputs \
    --search_mode prefix_match \
    --output_onnx_file_path ${FILE_NAME}.onnx
    sor4onnx \
    --input_onnx_file_path ${FILE_NAME}.onnx \
    --old_new "float_segments_raw_output___16:0" "float_segments_raw_output" \
    --mode full \
    --search_mode prefix_match \
    --output_onnx_file_path ${FILE_NAME}.onnx
    sor4onnx \
    --input_onnx_file_path ${FILE_NAME}.onnx \
    --old_new "float_segments_raw_output___12:0" "float_segments_raw_output" \
    --mode full \
    --search_mode prefix_match \
    --output_onnx_file_path ${FILE_NAME}.onnx
    sne4onnx \
    --input_onnx_file_path ${FILE_NAME}.onnx \
    --input_op_names input \
    --output_op_names heatmaps part_heatmaps float_segments_raw_output short_offsets \
    --output_onnx_file_path ${FILE_NAME}.onnx
}

MODEL=bodypix
MASK_SCORE_THRESHOLD=0.60
MODEL_TYPES=(
    "resnet50/stride16 16"
    "resnet50/stride32 32"
    "mobilenet050/stride8 8"
    "mobilenet050/stride16 16"
    "mobilenet075/stride8 8"
    "mobilenet075/stride16 16"
    "mobilenet100/stride8 8"
    "mobilenet100/stride16 16"
)
RESOLUTIONS=(
    "128 160"
    "128 256"
    "192 320"
    "192 416"
    "192 640"
    "192 800"
    "256 320"
    "256 416"
    "256 448"
    "256 640"
    "256 800"
    "256 960"
    "288 1280"
    "288 480"
    "288 640"
    "288 800"
    "288 960"
    "320 320"
    "384 1280"
    "384 480"
    "384 640"
    "384 800"
    "384 960"
    "416 416"
    "480 1280"
    "480 640"
    "480 800"
    "480 960"
    "512 512"
    "512 640"
    "512 896"
    "544 1280"
    "544 800"
    "544 960"
    "576 1024"
    "640 640"
    "736 1280"
    "160 96"
    "192 128"
    "224 160"
    "256 192"
    "384 288"
)

# ONNX export
for((i=0; i<${#MODEL_TYPES[@]}; i++))
do
    TYPE=(`echo ${MODEL_TYPES[i]}`)
    MODEL_TYPE=${TYPE[0]}
    STRIDES=${TYPE[1]}
    FILE_NAME="${MODEL}_${MODEL_TYPE//\//_}_1x3xHxW"
    onnx_export
done

# Fixed resolution
for((i=0; i<${#MODEL_TYPES[@]}; i++))
do
    TYPE=(`echo ${MODEL_TYPES[i]}`)
    MODEL_TYPE=${TYPE[0]}
    STRIDES=${TYPE[1]}

    for((j=0; j<${#RESOLUTIONS[@]}; j++))
    do
        RESOLUTION=(`echo ${RESOLUTIONS[j]}`)
        H=${RESOLUTION[0]}
        W=${RESOLUTION[1]}
        OH=$(( H / STRIDES ))
        OW=$(( W / STRIDES ))

        FILE_NAME="${MODEL}_${MODEL_TYPE//\//_}"

        # main
        onnxsim ${FILE_NAME}_1x3xHxW.onnx ${FILE_NAME}_1x3x${H}x${W}.onnx \
        --overwrite-input-shape "input:1,3,${H},${W}"

        sog4onnx \
        --op_type Transpose \
        --opset 11 \
        --op_name segment_trans \
        --input_variables segment_trans_input float32 [1,1,${OH},${OW}] \
        --attributes perm int64 [0,2,3,1] \
        --output_variables segments float32 [1,${OH},${OW},1]

        snc4onnx \
        --input_onnx_file_paths ${FILE_NAME}_1x3x${H}x${W}.onnx Transpose.onnx \
        --output_onnx_file_path ${FILE_NAME}_1x3x${H}x${W}.onnx \
        --srcop_destop float_segments_raw_output segment_trans_input

        # post-process
        python 01_make_mask.py -o 11 -b 1 -sh ${OH} -sw ${OW} -s ${STRIDES} -t ${MASK_SCORE_THRESHOLD}
        python 02_make_colored_mask.py -o 11 -b 1 -sh ${OH} -sw ${OW} -s ${STRIDES}
        python 04_make_pose_keypoints.py -o 11 -b 1 -sh ${OH} -sw ${OW} -s ${STRIDES}

        sor4onnx \
        --input_onnx_file_path 01_segment_mask_1x3x${H}x${W}.onnx \
        --old_new "/" "01/" \
        --mode full \
        --search_mode prefix_match \
        --output_onnx_file_path 01_segment_mask_1x3x${H}x${W}.onnx
        sor4onnx \
        --input_onnx_file_path 01_segment_mask_1x3x${H}x${W}.onnx \
        --old_new "onnx::" "01_" \
        --mode full \
        --search_mode prefix_match \
        --output_onnx_file_path 01_segment_mask_1x3x${H}x${W}.onnx

        sor4onnx \
        --input_onnx_file_path 02_colored_segment_mask_1x3x${H}x${W}.onnx \
        --old_new "/" "02/" \
        --mode full \
        --search_mode prefix_match \
        --output_onnx_file_path 02_colored_segment_mask_1x3x${H}x${W}.onnx
        sor4onnx \
        --input_onnx_file_path 02_colored_segment_mask_1x3x${H}x${W}.onnx \
        --old_new "onnx::" "02_" \
        --mode full \
        --search_mode prefix_match \
        --output_onnx_file_path 02_colored_segment_mask_1x3x${H}x${W}.onnx

        snc4onnx \
        --input_onnx_file_paths 01_segment_mask_1x3x${H}x${W}.onnx 02_colored_segment_mask_1x3x${H}x${W}.onnx \
        --output_onnx_file_path 03_segment_mask_colored_mask.onnx \
        --srcop_destop mask_for_colored_output mask_for_colored_input

        sor4onnx \
        --input_onnx_file_path 03_segment_mask_colored_mask.onnx \
        --old_new "01/Less" "mask_score_threshold_less" \
        --mode full \
        --search_mode prefix_match \
        --output_onnx_file_path 03_segment_mask_colored_mask.onnx

        sor4onnx \
        --input_onnx_file_path 03_segment_mask_colored_mask.onnx \
        --old_new "01/Constant_6_output_0" "mask_score_threshold" \
        --mode full \
        --search_mode prefix_match \
        --output_onnx_file_path 03_segment_mask_colored_mask.onnx

        onnxsim 03_segment_mask_colored_mask.onnx 03_segment_mask_colored_mask.onnx

        sor4onnx \
        --input_onnx_file_path 04_pose_keypoints_1x3x${H}x${W}.onnx \
        --old_new "/" "04/" \
        --mode full \
        --search_mode prefix_match \
        --output_onnx_file_path 04_pose_keypoints_1x3x${H}x${W}.onnx

        # merge
        snc4onnx \
        --input_onnx_file_paths ${FILE_NAME}_1x3x${H}x${W}.onnx 03_segment_mask_colored_mask.onnx \
        --output_onnx_file_path ${FILE_NAME}_1x3x${H}x${W}.onnx \
        --srcop_destop part_heatmaps part_heatmaps_input segments mask_input

        snc4onnx \
        --input_onnx_file_paths ${FILE_NAME}_1x3x${H}x${W}.onnx 04_pose_keypoints_1x3x${H}x${W}.onnx \
        --output_onnx_file_path ${FILE_NAME}_1x3x${H}x${W}.onnx \
        --srcop_destop heatmaps heatmaps_input short_offsets offsets_input

        onnxsim ${FILE_NAME}_1x3x${H}x${W}.onnx ${FILE_NAME}_1x3x${H}x${W}.onnx

        mv ${FILE_NAME}_1x3x${H}x${W}.onnx "saved_model_${MODEL_TYPE}/"
        rm Transpose.onnx
        rm 01_*.onnx
        rm 02_*.onnx
        rm 03_*.onnx
        rm 04_*.onnx
    done
    mv ${FILE_NAME}_1x3xHxW.onnx "saved_model_${MODEL_TYPE}/"
done










