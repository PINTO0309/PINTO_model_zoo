#!/bin/bash

pip install -U pip \
&& pip install onnxsim==0.4.17 \
&& pip install -U simple-onnx-processing-tools \
&& pip install onnx==1.13.1 \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
&& pip install -U onnx2tf


MODELS=(
    "directmhp_300wlp_m_finetune"
    "directmhp_300wlp_s_finetune"
    "directmhp_agora_m"
    "directmhp_agora_s"
    "directmhp_cmu_m"
    "directmhp_cmu_s"
)

RESOLUTIONS=(
    "192 320"
    "192 640"
    "256 320"
    "256 640"
    "256 960"
    "320 320"
    "384 640"
    "384 960"
    "384 1280"
    "512 512"
    "512 640"
    "512 960"
    "512 1280"
    "640 640"
    "768 1280"
)

for((i=0; i<${#MODELS[@]}; i++))
do
    MODEL=(`echo ${MODELS[i]}`)

    for((j=0; j<${#RESOLUTIONS[@]}; j++))
    do
        RESOLUTION=(`echo ${RESOLUTIONS[j]}`)
        H=${RESOLUTION[0]}
        W=${RESOLUTION[1]}
        echo @@@@@@@@@@@@@@@@@ processing ${MODEL}_${H}x${W} ...

        mkdir -p saved_model_${MODEL}_post_${H}x${W}
        onnx2tf -i withpost/${MODEL}_post_${H}x${W}.onnx -coion -o saved_model_${MODEL}_post_${H}x${W}
    done
done
