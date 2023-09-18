#!/bin/bash

TASKS=(
    "depth"
    "pose"
)

RESOLUTIONS=(
    "128 416"
    "192 320"
    "240 320"
    "256 448"
    "256 832"
    "360 640"
    "480 640"
    "512 896"
    "720 1280"
)

for((i=0; i<${#TASKS[@]}; i++))
do
    TASK=(`echo ${TASKS[i]}`)
    for((j=0; j<${#RESOLUTIONS[@]}; j++))
    do
        RESOLUTION=(`echo ${RESOLUTIONS[j]}`)
        H=${RESOLUTION[0]}
        W=${RESOLUTION[1]}

        printf "#### ${TASK}_${H}x${W}\n"
        python -m tf2onnx.convert --saved-model saved_model_${TASK}_${H}x${W} --output geonet_${TASK}_${H}x${W}.onnx --opset 11 --inputs-as-nchw input:0
        onnxsim geonet_${TASK}_${H}x${W}.onnx geonet_${TASK}_${H}x${W}.onnx
    done
done

TASK=flow
for((j=0; j<${#RESOLUTIONS[@]}; j++))
do
    RESOLUTION=(`echo ${RESOLUTIONS[j]}`)
    H=${RESOLUTION[0]}
    W=${RESOLUTION[1]}

    printf "#### ${TASK}_${H}x${W}\n"
    python -m tf2onnx.convert --saved-model saved_model_${TASK}_${H}x${W} --output geonet_${TASK}_${H}x${W}.onnx --opset 11 --inputs-as-nchw src_stack_input:0,tgt_input:0
    onnxsim geonet_${TASK}_${H}x${W}.onnx geonet_${TASK}_${H}x${W}.onnx
    onnxsim geonet_${TASK}_${H}x${W}.onnx geonet_${TASK}_${H}x${W}.onnx
    sne4onnx \
    --input_onnx_file_path geonet_${TASK}_${H}x${W}.onnx \
    --input_op_names src_image_stack tgt_image \
    --output_op_names fwd_full_flow_pyramid \
    --output_onnx_file_path geonet_${TASK}_${H}x${W}.onnx
done
