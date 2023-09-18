#!/bin/bash

ITERS=(
    "01"
    "05"
    "10"
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
    "512 640"
    "512 896"
    "544 800"
    "544 960"
    "544 1280"
    "640 640"
    "736 1280"
)

DATASET=kitti
MODEL=high_frequency_stereo_matching_${DATASET}

for((i=0; i<${#ITERS[@]}; i++))
do
    ITER=(`echo ${ITERS[i]}`)
    for((j=0; j<${#RESOLUTIONS[@]}; j++))
    do
        RESOLUTION=(`echo ${RESOLUTIONS[j]}`)
        H=${RESOLUTION[0]}
        W=${RESOLUTION[1]}

        printf "#### ${MODEL}_iter${ITER}_1x3x${H}x${W}\n"
        spo4onnx ${MODEL}_iter${ITER}_1x3xHxW.onnx ${MODEL}_iter${ITER}_1x3x${H}x${W}.onnx --overwrite-input-shape "left:1,3,${H},${W}" "right:1,3,${H},${W}"
    done
done
