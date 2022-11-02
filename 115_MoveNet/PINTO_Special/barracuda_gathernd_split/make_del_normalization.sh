#!/bin/bash

PARAMLIST=(
    "192 256 6"
    "192 256 10"
    "192 256 20"
    "192 320 6"
    "192 320 10"
    "192 320 20"
    "256 320 6"
    "256 320 10"
    "256 320 20"
    "256 416 6"
    "256 416 10"
    "256 416 20"
    "288 480 6"
    "288 480 10"
    "288 480 20"
    "384 640 6"
    "384 640 10"
    "384 640 20"
    "384 1280 6"
    "384 1280 10"
    "384 1280 20"
    "480 640 6"
    "480 640 10"
    "480 640 20"
    "480 800 6"
    "480 800 10"
    "480 800 20"
    "736 1280 6"
    "736 1280 10"
    "736 1280 20"
)
for((i=0; i<${#PARAMLIST[@]}; i++))
do
    PARAM=(`echo ${PARAMLIST[i]}`)
    H=${PARAM[0]}
    W=${PARAM[1]}
    P=${PARAM[2]}

    snd4onnx \
    --remove_node_names StatefulPartitionedCall/truediv_1 StatefulPartitionedCall/sub_1 \
    --input_onnx_file_path movenet_multipose_lightning_${H}x${W}_p${P}_nopost_myriad_barracuda.onnx \
    --output_onnx_file_path movenet_multipose_lightning_${H}x${W}_p${P}_nopost_barracuda_for_Texture2D.onnx
done
