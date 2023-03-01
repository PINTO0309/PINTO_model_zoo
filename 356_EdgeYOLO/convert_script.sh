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
    "192 320 1260 80"
    "192 416 1638 80"
    "192 640 2520 80"
    "192 800 3150 80"
    "256 320 1680 80"
    "256 416 2184 80"
    "256 640 3360 80"
    "256 800 4200 80"
    "256 960 5040 80"
    "288 480 2835 80"
    "288 640 3780 80"
    "288 800 4725 80"
    "288 960 5670 80"
    "288 1280 7560 80"
    "320 320 2100 80"
    "384 480 3780 80"
    "384 640 5040 80"
    "384 800 6300 80"
    "384 960 7560 80"
    "384 1280 10080 80"
    "416 416 3549 80"
    "480 640 6300 80"
    "480 800 7875 80"
    "480 960 9450 80"
    "480 1280 12600 80"
    "512 512 5376 80"
    "544 800 8925 80"
    "544 960 10710 80"
    "544 1280 14280 80"
    "640 640 8400 80"
    "736 1280 19320 80"
)

################################################## Base Component gen
onnx2json \
--input_onnx_file_path nms_base_component.onnx \
--output_json_path nms_base_component.json \
--json_indent 2

################################################## Post-Process + NMS gen
for((i=0; i<${#RESOLUTIONS[@]}; i++))
do
    RESOLUTION=(`echo ${RESOLUTIONS[i]}`)
    H=${RESOLUTION[0]}
    W=${RESOLUTION[1]}
    BOXES=${RESOLUTION[2]}
    CLASSES=${RESOLUTION[3]}

    python make_yolo_postprocess.py \
    --opset 11 \
    --model_input_shape 1 3 ${H} ${W} \
    --strides 8 16 32 \
    --classes ${CLASSES} \
    --boxes ${BOXES}

    cp nms_base_component.json postprocess_${BOXES}.json
    sed -i -e 's/"1260"/'$BOXES'/g' postprocess_${BOXES}.json
    sed -i -e 's/"80"/'$CLASSES'/g' postprocess_${BOXES}.json
    json2onnx \
    --input_json_path postprocess_${BOXES}.json \
    --output_onnx_file_path postprocess_${BOXES}.onnx
    rm postprocess_${BOXES}.json

    snc4onnx \
    --input_onnx_file_paths postprocess_anchors_${BOXES}.onnx postprocess_${BOXES}.onnx \
    --output_onnx_file_path postprocess_${BOXES}.onnx \
    --srcop_destop bboxes_xyxy post_boxes scores post_scores

    rm postprocess_anchors_${BOXES}.onnx
done


################################################## Model + Post-Process merge
for((i=0; i<${#WEIGHTS[@]}; i++))
do
    MODEL=(`echo ${WEIGHTS[i]}`)
    for((j=0; j<${#RESOLUTIONS[@]}; j++))
    do
        RESOLUTION=(`echo ${RESOLUTIONS[j]}`)
        H=${RESOLUTION[0]}
        W=${RESOLUTION[1]}
        BOXES=${RESOLUTION[2]}
        CLASSES=${RESOLUTION[3]}
        echo @@@@@@@@@@@@@@@@@ processing ${MODEL}_${H}x${W} ...
        snc4onnx \
        --input_onnx_file_paths ${MODEL}/${MODEL}_${H}x${W}.onnx postprocess_${BOXES}.onnx \
        --output_onnx_file_path ${MODEL}/${MODEL}_${H}x${W}_post.onnx \
        --srcop_destop output_0 post_input
    done
done