#!/bin/bash

PARAMLIST=(
    "192 256 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "192 256 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/stack_1_Concat__512 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "192 256 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "192 320 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "192 320 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "192 320 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "256 320 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "256 320 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "256 320 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "256 416 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "256 416 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "256 416 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "288 480 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "288 480 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "288 480 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "384 640 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "384 640 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "384 640 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "384 1280 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "384 1280 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "384 1280 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "480 640 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "480 640 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "480 640 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "480 800 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "480 800 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "480 800 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
    "736 1280 6 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "736 1280 10 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd__522 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_3__586 StatefulPartitionedCall/GatherNd_4"
    "736 1280 20 StatefulPartitionedCall/box_scale_0/conv2d_5/BiasAdd__521 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd StatefulPartitionedCall/box_offset_0/conv2d_6/BiasAdd__536 StatefulPartitionedCall/GatherNd_1__537 StatefulPartitionedCall/GatherNd_1 StatefulPartitionedCall/kpt_regress_0/conv2d_8/BiasAdd__442 StatefulPartitionedCall/GatherNd_2__495 StatefulPartitionedCall/GatherNd_2 StatefulPartitionedCall/Reshape_8 StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_3 StatefulPartitionedCall/Max StatefulPartitionedCall/GatherNd_4__593 StatefulPartitionedCall/GatherNd_4"
)
for((i=0; i<${#PARAMLIST[@]}; i++))
do
    PARAM=(`echo ${PARAMLIST[i]}`)
    H=${PARAM[0]}
    W=${PARAM[1]}
    P=${PARAM[2]}

    GATHERND0_TRANSPOSE=${PARAM[3]}
    GATHERND0_CAST=${PARAM[4]}
    GATHERND0_RESHAPE=${PARAM[5]}
    GATHERND1_TRANSPOSE=${PARAM[6]}
    GATHERND1_CAST=${PARAM[7]}
    GATHERND1_RESHAPE=${PARAM[8]}
    GATHERND2_TRANSPOSE=${PARAM[9]}
    GATHERND2_CAST=${PARAM[10]}
    GATHERND2_RESHAPE=${PARAM[11]}
    GATHERND3_RESHAPE=${PARAM[12]}
    GATHERND3_CAST=${PARAM[13]}
    GATHERND3_SPLIT=${PARAM[14]}
    GATHERND4_REDUCEMAX=${PARAM[15]}
    GATHERND4_CAST=${PARAM[16]}
    GATHERND4_RESHAPE=${PARAM[17]}

    echo @@@@@@@@@@@@@@@@@ processing ${H}x${W} p=${P} ...

    cp 192x256_p6/* .

    ############################################################################################### GatherND
    QUATH=$((${H} / 4))
    QUATW=$((${W} / 4))
    NEWSHAPE1=$((${QUATH} * ${QUATW}))
    NEWSHAPE2=$((${QUATH} * ${QUATW} * 17))
    NUM=0
    MULVAL0=`sed4onnx --constant_string [${NEWSHAPE1},${QUATW},1] --dtype int64 --mode encode`
    onnx2json \
    --input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
    --output_json_path barracuda_gather_nd_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"rawData\": \"AAwAAAAAAABAAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL0}\"/g" barracuda_gather_nd_${NUM}.json
    json2onnx \
    --input_json_path barracuda_gather_nd_${NUM}.json \
    --output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
    rm barracuda_gather_nd_${NUM}.json

    NUM=1
    MULVAL1=`sed4onnx --constant_string [${NEWSHAPE1},${QUATW},1] --dtype int64 --mode encode`
    onnx2json \
    --input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
    --output_json_path barracuda_gather_nd_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"rawData\": \"AAwAAAAAAABAAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL1}\"/g" barracuda_gather_nd_${NUM}.json
    json2onnx \
    --input_json_path barracuda_gather_nd_${NUM}.json \
    --output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
    rm barracuda_gather_nd_${NUM}.json

    NUM=2
    MULVAL2=`sed4onnx --constant_string [${QUATW},1] --dtype int64 --mode encode`
    onnx2json \
    --input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
    --output_json_path barracuda_gather_nd_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"3072\"/\"dimValue\": \"${NEWSHAPE1}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"rawData\": \"QAAAAAAAAAABAAAAAAAAAA==\"/\"rawData\": \"${MULVAL2}\"/g" barracuda_gather_nd_${NUM}.json
    json2onnx \
    --input_json_path barracuda_gather_nd_${NUM}.json \
    --output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
    rm barracuda_gather_nd_${NUM}.json

    NUM=3
    MULVAL3=`sed4onnx --constant_string [$((${NEWSHAPE2} / ${QUATH})),17,1] --dtype int64 --mode encode`
    onnx2json \
    --input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
    --output_json_path barracuda_gather_nd_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"52224\"/\"dimValue\": \"${NEWSHAPE2}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"rawData\": \"QAQAAAAAAAARAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL3}\"/g" barracuda_gather_nd_${NUM}.json
    json2onnx \
    --input_json_path barracuda_gather_nd_${NUM}.json \
    --output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
    rm barracuda_gather_nd_${NUM}.json

    NUM=4
    MULVAL4=`sed4onnx --constant_string [$((${NEWSHAPE2} / ${QUATH})),17,1] --dtype int64 --mode encode`
    onnx2json \
    --input_onnx_file_path barracuda_gather_nd_${NUM}.onnx \
    --output_json_path barracuda_gather_nd_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"48\"/\"dimValue\": \"${QUATH}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"64\"/\"dimValue\": \"${QUATW}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"dimValue\": \"52224\"/\"dimValue\": \"${NEWSHAPE2}\"/g" barracuda_gather_nd_${NUM}.json
    sed -i -e "s/\"rawData\": \"QAQAAAAAAAARAAAAAAAAAAEAAAAAAAAA\"/\"rawData\": \"${MULVAL4}\"/g" barracuda_gather_nd_${NUM}.json
    json2onnx \
    --input_json_path barracuda_gather_nd_${NUM}.json \
    --output_onnx_file_path barracuda_gather_nd_${NUM}.onnx
    rm barracuda_gather_nd_${NUM}.json


    ############################################################################################### Split
    NUM=0
    onnx2json \
    --input_onnx_file_path barracuda_split_${NUM}.onnx \
    --output_json_path barracuda_split_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
    json2onnx \
    --input_json_path barracuda_split_${NUM}.json \
    --output_onnx_file_path barracuda_split_${NUM}.onnx
    rm barracuda_split_${NUM}.json
    NUM=1
    onnx2json \
    --input_onnx_file_path barracuda_split_${NUM}.onnx \
    --output_json_path barracuda_split_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
    json2onnx \
    --input_json_path barracuda_split_${NUM}.json \
    --output_onnx_file_path barracuda_split_${NUM}.onnx
    rm barracuda_split_${NUM}.json
    NUM=2
    onnx2json \
    --input_onnx_file_path barracuda_split_${NUM}.onnx \
    --output_json_path barracuda_split_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
    json2onnx \
    --input_json_path barracuda_split_${NUM}.json \
    --output_onnx_file_path barracuda_split_${NUM}.onnx
    rm barracuda_split_${NUM}.json
    NUM=3
    onnx2json \
    --input_onnx_file_path barracuda_split_${NUM}.onnx \
    --output_json_path barracuda_split_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"6\"/\"dimValue\": \"${P}\"/g" barracuda_split_${NUM}.json
    json2onnx \
    --input_json_path barracuda_split_${NUM}.json \
    --output_onnx_file_path barracuda_split_${NUM}.onnx
    rm barracuda_split_${NUM}.json
    NUM=4
    onnx2json \
    --input_onnx_file_path barracuda_split_${NUM}.onnx \
    --output_json_path barracuda_split_${NUM}.json \
    --json_indent 2
    sed -i -e "s/\"dimValue\": \"102\"/\"dimValue\": \"$((${P} * 17))\"/g" barracuda_split_${NUM}.json
    json2onnx \
    --input_json_path barracuda_split_${NUM}.json \
    --output_onnx_file_path barracuda_split_${NUM}.onnx
    rm barracuda_split_${NUM}.json

    ############################################################################################### Merge Process
    MODEL=movenet_multipose_lightning_${H}x${W}_p${P}_nopost_myriad
    echo 001
    sor4onnx \
    --input_onnx_file_path ${MODEL}.onnx \
    --old_new "${GATHERND0_TRANSPOSE}" "gnd0_Transpose" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx
    echo 002
    sor4onnx \
    --input_onnx_file_path ${MODEL}_barracuda.onnx \
    --old_new "${GATHERND0_CAST}" "gnd01_Cast" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx
    echo 003
    sor4onnx \
    --input_onnx_file_path ${MODEL}_barracuda.onnx \
    --old_new "${GATHERND1_TRANSPOSE}" "gnd1_Transpose" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx

    echo 004
    sor4onnx \
    --input_onnx_file_path ${MODEL}_barracuda.onnx \
    --old_new "${GATHERND2_TRANSPOSE}" "gnd2_Transpose" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx
    echo 005
    sor4onnx \
    --input_onnx_file_path ${MODEL}_barracuda.onnx \
    --old_new "${GATHERND2_CAST}" "gnd2_Cast" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx

    echo 006
    sor4onnx \
    --input_onnx_file_path ${MODEL}_barracuda.onnx \
    --old_new "${GATHERND3_RESHAPE}" "gnd3_Reshape" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx
    echo 007
    sor4onnx \
    --input_onnx_file_path ${MODEL}_barracuda.onnx \
    --old_new "${GATHERND3_CAST}" "gnd34_Cast" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx
    echo 008
    sor4onnx \
    --input_onnx_file_path ${MODEL}_barracuda.onnx \
    --old_new "${GATHERND4_REDUCEMAX}" "gnd4_ReduceMax" \
    --output_onnx_file_path ${MODEL}_barracuda.onnx


    ###################################################################################
    MODEL2=${MODEL}_barracuda
    NUM=0
    echo 009
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
    --srcop_destop gnd0_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 010
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "bgn${NUM}_output" \
    --to_input_variable_name "${GATHERND0_RESHAPE}" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 011
    snd4onnx \
    --remove_node_names ${GATHERND0_RESHAPE} \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=1
    echo 012
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
    --srcop_destop gnd1_Transpose bgn${NUM}_data gnd01_Cast bgn${NUM}_indices \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 013
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "bgn${NUM}_output" \
    --to_input_variable_name "${GATHERND1_RESHAPE}" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 014
    snd4onnx \
    --remove_node_names ${GATHERND1_RESHAPE} \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=2
    echo 015
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
    --srcop_destop gnd2_Transpose bgn${NUM}_data gnd2_Cast bgn${NUM}_indices \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 016
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "bgn${NUM}_output" \
    --to_input_variable_name "${GATHERND2_RESHAPE}" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 017
    snd4onnx \
    --remove_node_names ${GATHERND2_RESHAPE} \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=3
    echo 018
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
    --srcop_destop gnd3_Reshape bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 019
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "bgn${NUM}_output" \
    --to_input_variable_name "${GATHERND3_SPLIT}" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 020
    snd4onnx \
    --remove_node_names ${GATHERND3_SPLIT} \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=4
    echo 021
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_gather_nd_${NUM}.onnx \
    --srcop_destop gnd4_ReduceMax bgn${NUM}_data gnd34_Cast bgn${NUM}_indices \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 022
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "bgn${NUM}_output" \
    --to_input_variable_name "${GATHERND4_RESHAPE}" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 023
    snd4onnx \
    --remove_node_names ${GATHERND4_RESHAPE} \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx


    ###################################################################################
    MODEL2=${MODEL}_barracuda
    NUM=0
    echo 024
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
    --srcop_destop Max__524 barracuda_split_${NUM}_input \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 025
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split0_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 026
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split1_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack:1" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 027
    snd4onnx \
    --remove_node_names StatefulPartitionedCall/unstack \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=1
    echo 028
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
    --srcop_destop StatefulPartitionedCall/Reshape_7 barracuda_split_${NUM}_input \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 029
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split0_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack_1" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 030
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split1_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack_1:1" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 031
    snd4onnx \
    --remove_node_names StatefulPartitionedCall/unstack_1 \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=2
    echo 032
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
    --srcop_destop StatefulPartitionedCall/Reshape_9 barracuda_split_${NUM}_input \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 033
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split0_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack_2" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 034
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split1_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack_2:1" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 035
    snd4onnx \
    --remove_node_names StatefulPartitionedCall/unstack_2 \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=3
    echo 036
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
    --srcop_destop StatefulPartitionedCall/Squeeze_4 barracuda_split_${NUM}_input \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 037
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split0_output" \
    --to_input_variable_name "StatefulPartitionedCall/split" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 038
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split1_output" \
    --to_input_variable_name "StatefulPartitionedCall/split:1" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 039
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split2_output" \
    --to_input_variable_name "StatefulPartitionedCall/split:2" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 040
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split3_output" \
    --to_input_variable_name "StatefulPartitionedCall/split:3" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 041
    snd4onnx \
    --remove_node_names StatefulPartitionedCall/split \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    NUM=4
    echo 042
    snc4onnx \
    --input_onnx_file_paths ${MODEL2}.onnx barracuda_split_${NUM}.onnx \
    --srcop_destop bgn3_output barracuda_split_${NUM}_input \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 043
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split0_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack_3" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 044
    svs4onnx \
    --input_onnx_file_path ${MODEL2}.onnx \
    --from_output_variable_name "barracuda_split_${NUM}_split1_output" \
    --to_input_variable_name "StatefulPartitionedCall/unstack_3:1" \
    --output_onnx_file_path ${MODEL2}.onnx
    echo 045
    snd4onnx \
    --remove_node_names StatefulPartitionedCall/unstack_3 \
    --input_onnx_file_path ${MODEL2}.onnx \
    --output_onnx_file_path ${MODEL2}.onnx

    ###################################################################################
    mkdir -p ${H}x${W}_p${P}
    mv barracuda_gather_nd_*.onnx ${H}x${W}_p${P}
    mv barracuda_split_*.onnx ${H}x${W}_p${P}
done