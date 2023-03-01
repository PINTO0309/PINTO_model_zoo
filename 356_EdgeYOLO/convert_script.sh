#!/bin/bash

WEIGHTS=(
    "edgeyolo_coco"
    "edgeyolo_m_coco"
    "edgeyolo_m_visdrone"
    "edgeyolo_s_coco"
    "edgeyolo_s_visdrone"
    "edgeyolo_tiny_coco"
    "edgeyolo_tiny_lrelu_coco"
    "edgeyolo_tiny_lrelu_visdrone"
    "edgeyolo_tiny_visdrone"
    "edgeyolo_visdrone"
)
RESOLUTIONS=(
    "192 320"
    "192 416"
    "192 640"
    "192 800"
    "256 320"
    "256 416"
    "256 640"
    "256 800"
    "256 960"
    "288 480"
    "288 640"
    "288 800"
    "288 960"
    "288 1280"
    "320 320"
    "384 480"
    "384 640"
    "384 800"
    "384 960"
    "384 1280"
    "416 416"
    "480 640"
    "480 800"
    "480 960"
    "480 1280"
    "512 512"
    "544 800"
    "544 960"
    "544 1280"
    "640 640"
    "736 1280"
)

for((i=0; i<${#WEIGHTS[@]}; i++))
do
    WEIGHT=(`echo ${WEIGHTS[i]}`)
    for((j=0; j<${#RESOLUTIONS[@]}; j++))
    do
        RESOLUTION=(`echo ${RESOLUTIONS[j]}`)
        H=${RESOLUTION[0]}
        W=${RESOLUTION[1]}
        MODELNAME=${WEIGHT}_${H}x${W}
        echo @@@@@@@@@@@@@@@@@ processing ${MODELNAME} ...

        python export.py \
        --onnx-only \
        --weights weights/${WEIGHT}.pth \
        --input-size ${H} ${W} \
        --batch 1 \
        --opset 11
    done
done



WEIGHTS=(
    "edgeyolo_coco"
    "edgeyolo_m_coco"
    "edgeyolo_m_visdrone"
    "edgeyolo_s_coco"
    "edgeyolo_s_visdrone"
    "edgeyolo_tiny_coco"
    "edgeyolo_tiny_lrelu_coco"
    "edgeyolo_tiny_lrelu_visdrone"
    "edgeyolo_tiny_visdrone"
    "edgeyolo_visdrone"
)
RESOLUTIONS=(
    "192 320"
)

for((i=0; i<${#WEIGHTS[@]}; i++))
do
    WEIGHT=(`echo ${WEIGHTS[i]}`)
    for((j=0; j<${#RESOLUTIONS[@]}; j++))
    do
        RESOLUTION=(`echo ${RESOLUTIONS[j]}`)
        H=${RESOLUTION[0]}
        W=${RESOLUTION[1]}
        MODELNAME=${WEIGHT}_${H}x${W}
        echo @@@@@@@@@@@@@@@@@ processing ${MODELNAME} ...

        python export.py \
        --onnx-only \
        --weights weights/${WEIGHT}.pth \
        --input-size ${H} ${W} \
        --batch 1 \
        --opset 11 \
        --dynamic
    done
done