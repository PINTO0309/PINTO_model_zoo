#!/bin/bash

FRAMES=(
    "27"
    "81"
    "243"
    "351"
)

for((i=0; i<${#FRAMES[@]}; i++))
do
    FRAME=(`echo ${FRAMES[i]}`)
    MODELNAME=mhformer_NxFxKxXY_1x${FRAME}x17x2
    echo @@@@@@@@@@@@@@@@@ processing ${MODELNAME} ...
    onnx2tf -i ${MODELNAME}.onnx -o ${MODELNAME} -osd
done