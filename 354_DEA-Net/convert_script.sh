#!/bin/bash

DATASETS=(
    "haze4k"
    "its"
    "ots"
)
RESOLUTIONS=(
    "180 320"
    "180 416"
    "180 640"
    "180 800"
    "240 320"
    "240 416"
    "240 640"
    "240 800"
    "240 960"
    "288 480"
    "288 640"
    "288 800"
    "288 960"
    "288 1280"
    "320 320"
    "360 480"
    "360 640"
    "360 800"
    "360 960"
    "360 1280"
    "416 416"
    "480 640"
    "480 800"
    "480 960"
    "480 1280"
    "540 800"
    "540 960"
    "540 1280"
    "640 640"
    "720 1280"
)

for((i=0; i<${#DATASETS[@]}; i++))
do
    DATASET=(`echo ${DATASETS[i]}`)
    for((k=0; k<${#RESOLUTIONS[@]}; k++))
    do
        RESOLUTION=(`echo ${RESOLUTIONS[k]}`)
        H=${RESOLUTION[0]}
        W=${RESOLUTION[1]}
        MODELNAME=dea_net_${DATASET}_${H}x${W}
        echo @@@@@@@@@@@@@@@@@ processing ${MODELNAME} ...
        onnx2tf -i ${MODELNAME}.onnx -o ${MODELNAME}
    done
done
