#!/bin/bash

# https://zenn.dev/pinto0309/scraps/63167cc8709f0b
# https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rtdetr

# pip install onnx==1.13.1
# pip install paddle2onnx==1.0.5
# pip install onnxsim==0.4.17

git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection/
# git checkout develop
git checkout 72ae571da36aff55058c70f302e57097ad6278fc

MODELS=(
    "rtdetr_r50vd_6x_coco"
    "rtdetr_r101vd_6x_coco"
    "rtdetr_hgnetv2_l_6x_coco"
    "rtdetr_hgnetv2_x_6x_coco"
)

RESOLUTIONS=(
    "192 320"
    "192 416"
    "192 512"
    "192 640"
    "192 800"
    "256 320"
    "256 416"
    "256 512"
    "256 640"
    "256 800"
    "256 960"
    "288 480"
    "288 512"
    "288 640"
    "288 800"
    "288 960"
    "288 1280"
    "320 320"
    "384 480"
    "384 512"
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

wget https://github.com/jiangjiajun/PaddleUtils/raw/main/paddle/paddle_infer_shape.py

for((i=0; i<${#MODELS[@]}; i++))
do
    MODEL=(`echo ${MODELS[i]}`)

    for((j=0; j<${#RESOLUTIONS[@]}; j++))
    do
        RESOLUTION=(`echo ${RESOLUTIONS[j]}`)
        H=${RESOLUTION[0]}
        W=${RESOLUTION[1]}
        echo @@@@@@@@@@@@@@@@@ processing ${MODEL}_${H}x${W} ...

        sed -i -e \
        "s/\[640, 640\]/\[${H}, ${W}\]/g" \
        configs/rtdetr/_base_/rtdetr_r50vd.yml

        sed -i -e \
        "s/\[3, 640, 640\]/\[3, ${H}, ${W}\]/g" \
        configs/rtdetr/_base_/rtdetr_reader.yml

        sed -i -e \
        "s/\[640, 640\]/\[${H}, ${W}\]/g" \
        configs/rtdetr/_base_/rtdetr_reader.yml

        python tools/export_model.py \
        -c configs/rtdetr/${MODEL}.yml \
        -o weights=https://bj.bcebos.com/v1/paddledet/models/${MODEL}.pdparams trt=True use_gpu=False \
        --output_dir=output_inference

        python paddle_infer_shape.py \
        --model_dir=output_inference/${MODEL}/ \
        --model_filename model.pdmodel  \
        --params_filename model.pdiparams \
        --save_dir ${MODEL} \
        --input_shape_dict="{'image':[1, 3, ${H}, ${W}], 'im_shape': [1, 2], 'scale_factor': [1, 2]}"

        paddle2onnx \
        --model_dir=${MODEL}/ \
        --model_filename model.pdmodel  \
        --params_filename model.pdiparams \
        --opset_version 16 \
        --save_file ${MODEL}_${H}x${W}.onnx

        onnxsim ${MODEL}_${H}x${W}.onnx ${MODEL}_${H}x${W}.onnx
        onnxsim ${MODEL}_${H}x${W}.onnx ${MODEL}_${H}x${W}.onnx
        onnxsim ${MODEL}_${H}x${W}.onnx ${MODEL}_${H}x${W}.onnx

        sed -i -e \
        "s/\[${H}, ${W}\]/\[640, 640\]/g" \
        configs/rtdetr/_base_/rtdetr_r50vd.yml

        sed -i -e \
        "s/\[3, ${H}, ${W}\]/\[3, 640, 640\]/g" \
        configs/rtdetr/_base_/rtdetr_reader.yml

        sed -i -e \
        "s/\[${H}, ${W}\]/\[640, 640\]/g" \
        configs/rtdetr/_base_/rtdetr_reader.yml
    done
done


