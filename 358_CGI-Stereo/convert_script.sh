#!/bin/bash

DATASETS=(
    "kitti"
    "sceneflow"
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

for((i=0; i<${#DATASETS[@]}; i++))
do
    DATASET=(`echo ${DATASETS[i]}`)
    for((j=0; j<${#RESOLUTIONS[@]}; j++))
    do
        RESOLUTION=(`echo ${RESOLUTIONS[j]}`)
        H=${RESOLUTION[0]}
        W=${RESOLUTION[1]}
        MODELNAME=${DATASET}_${H}x${W}
        echo @@@@@@@@@@@@@@@@@ processing ${MODELNAME} ...

        onnx2tf \
        -i cgi_stereo_${MODELNAME}.onnx \
        -o cgi_stereo_${MODELNAME} \
        -prf replace_cgi_stereo.json

        mv cgi_stereo_${MODELNAME}.onnx cgi_stereo_${MODELNAME}
    done
done
