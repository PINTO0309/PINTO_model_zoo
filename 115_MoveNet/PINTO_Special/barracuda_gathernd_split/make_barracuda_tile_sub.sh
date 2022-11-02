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

    echo @@@@@@@@@@@@@@@@@ processing ${H}x${W} p=${P} ...

    echo 001
    NUM=0
    OPNAME=barracuda_tile_${NUM}
    sog4onnx \
    --op_type Tile \
    --opset 11 \
    --op_name ${OPNAME} \
    --input_variables ${OPNAME}_input float32 [1,1,${P},17] \
    --input_variables ${OPNAME}_repeats int64 [4] \
    --output_variables ${OPNAME}_output float32 [$((${H} / 4)),$((${W} / 4)),${P},17] \
    --output_onnx_file_path barracuda_tile_${H}x${W}_p${P}_${NUM}.onnx

    echo 002
    sog4onnx \
    --op_type Constant \
    --opset 11 \
    --op_name ${OPNAME}_const \
    --output_variables ${OPNAME}_repeats_const int64 [4] \
    --attributes value int64 [$((${H} / 4)),$((${W} / 4)),1,1] \
    --output_onnx_file_path barracuda_tile_${H}x${W}_p${P}_${NUM}_const.onnx

    echo 003
    snc4onnx \
    --input_onnx_file_paths barracuda_tile_${H}x${W}_p${P}_${NUM}.onnx barracuda_tile_${H}x${W}_p${P}_${NUM}_const.onnx \
    --srcop_destop ${OPNAME}_repeats_const ${OPNAME}_repeats \
    --output_onnx_file_path barracuda_tile_${H}x${W}_p${P}_${NUM}.onnx

    echo 004
    rm barracuda_tile_${H}x${W}_p${P}_${NUM}_const.onnx

    echo 005
    python make_broadcast_sub.py \
    --model_name_suffix ${NUM} \
    --persons ${P} \
    --sub_a_constat_path $((${H} / 4))x$((${W} / 4))_${NUM}.npy \
    --sub_b_shape $((${H} / 4)) $((${W} / 4)) ${P} 17 \
    --sub_b_data_type float32

    rm $((${H} / 4))x$((${W} / 4))_${NUM}.npy

    echo 006
    snc4onnx \
    --input_onnx_file_paths barracuda_tile_${H}x${W}_p${P}_${NUM}.onnx barracuda_broadcast_sub_${NUM}_${H}x${W}_p${P}.onnx \
    --srcop_destop ${OPNAME}_output barracuda_broadcast_sub_${NUM}_b \
    --output_onnx_file_path barracuda_tile_sub_${H}x${W}_p${P}_${NUM}.onnx

    rm barracuda_broadcast_sub_${NUM}_${H}x${W}_p${P}.onnx
    rm barracuda_tile_${H}x${W}_p${P}_${NUM}.onnx

    #####################################################################################

    echo 007
    NUM=1
    OPNAME=barracuda_tile_${NUM}
    sog4onnx \
    --op_type Tile \
    --opset 11 \
    --op_name ${OPNAME} \
    --input_variables ${OPNAME}_input float32 [1,1,${P},17] \
    --input_variables ${OPNAME}_repeats int64 [4] \
    --output_variables ${OPNAME}_output float32 [$((${H} / 4)),$((${W} / 4)),${P},17] \
    --output_onnx_file_path barracuda_tile_${H}x${W}_p${P}_${NUM}.onnx

    echo 008
    sog4onnx \
    --op_type Constant \
    --opset 11 \
    --op_name ${OPNAME}_const \
    --output_variables ${OPNAME}_repeats_const int64 [4] \
    --attributes value int64 [$((${H} / 4)),$((${W} / 4)),1,1] \
    --output_onnx_file_path barracuda_tile_${H}x${W}_p${P}_${NUM}_const.onnx

    echo 009
    snc4onnx \
    --input_onnx_file_paths barracuda_tile_${H}x${W}_p${P}_${NUM}.onnx barracuda_tile_${H}x${W}_p${P}_${NUM}_const.onnx \
    --srcop_destop ${OPNAME}_repeats_const ${OPNAME}_repeats \
    --output_onnx_file_path barracuda_tile_${H}x${W}_p${P}_${NUM}.onnx

    echo 010
    rm barracuda_tile_${H}x${W}_p${P}_${NUM}_const.onnx

    echo 011
    python make_broadcast_sub.py \
    --model_name_suffix ${NUM} \
    --persons ${P} \
    --sub_a_constat_path $((${H} / 4))x$((${W} / 4))_${NUM}.npy \
    --sub_b_shape $((${H} / 4)) $((${W} / 4)) ${P} 17 \
    --sub_b_data_type float32

    rm $((${H} / 4))x$((${W} / 4))_${NUM}.npy

    echo 012
    snc4onnx \
    --input_onnx_file_paths barracuda_tile_${H}x${W}_p${P}_${NUM}.onnx barracuda_broadcast_sub_${NUM}_${H}x${W}_p${P}.onnx \
    --srcop_destop ${OPNAME}_output barracuda_broadcast_sub_${NUM}_b \
    --output_onnx_file_path barracuda_tile_sub_${H}x${W}_p${P}_${NUM}.onnx

    rm barracuda_broadcast_sub_${NUM}_${H}x${W}_p${P}.onnx
    rm barracuda_tile_${H}x${W}_p${P}_${NUM}.onnx
done
