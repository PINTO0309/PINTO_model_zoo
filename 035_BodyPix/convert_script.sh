#!/bin/bash

# pip install tensorflowjs -U --no-deps
# pip install openvino2tensorflow -U --no-deps
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
}

MODEL=bodypix
MODEL_TYPES=(
    "resnet50/stride16"
    "resnet50/stride32"
    "mobilenet050/stride8"
    "mobilenet050/stride16"
    "mobilenet075/stride8"
    "mobilenet075/stride16"
    "mobilenet100/stride8"
    "mobilenet100/stride16"
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
)

# ONNX export
for((i=0; i<${#MODEL_TYPES[@]}; i++))
do
    MODEL_TYPE=(`echo ${MODEL_TYPES[i]}`)
    FILE_NAME="${MODEL}_${MODEL_TYPE//\//_}_1x3xHxW"
    onnx_export
done

# Fixed resolution
for((i=0; i<${#MODEL_TYPES[@]}; i++))
do
    MODEL_TYPE=(`echo ${MODEL_TYPES[i]}`)

    for((j=0; j<${#RESOLUTIONS[@]}; j++))
    do
        RESOLUTION=(`echo ${RESOLUTIONS[j]}`)
        H=${RESOLUTION[0]}
        W=${RESOLUTION[1]}

        FILE_NAME="${MODEL}_${MODEL_TYPE//\//_}"

        onnxsim ${FILE_NAME}_1x3xHxW.onnx ${FILE_NAME}_1x${H}x${W}.onnx \
        --overwrite-input-shape "input:1,3,${H},${W}"

        mv ${FILE_NAME}_1x${H}x${W}.onnx saved_model_${MODEL_TYPE}/
    done
    mv ${FILE_NAME}_1x3xHxW.onnx saved_model_${MODEL_TYPE}/
done










