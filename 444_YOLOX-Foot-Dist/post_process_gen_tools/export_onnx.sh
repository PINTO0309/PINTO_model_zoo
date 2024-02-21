TYPE=m
MODEL_NAME=yolox_m_body_head_hand
SUFFIX="0299_0.5263_1x3x"
MODEL_PATH=yolox_outputs_m/YOLOX_outputs/m/best_ckpt_0299_0.5263.pth

RESOLUTIONS=(
    "128 160 420"
    "128 256 672"
    "192 320 1260"
    "192 416 1638"
    "192 640 2520"
    "192 800 3150"
    "256 320 1680"
    "256 416 2184"
    "256 448 2352"
    "256 640 3360"
    "256 800 4200"
    "256 960 5040"
    "288 1280 7560"
    "288 480 2835"
    "288 640 3780"
    "288 800 4725"
    "288 960 5670"
    "320 320 2100"
    "384 1280 10080"
    "384 480 3780"
    "384 640 5040"
    "384 800 6300"
    "384 960 7560"
    "416 416 3549"
    "480 1280 12600"
    "480 640 6300"
    "480 800 7875"
    "480 960 9450"
    "512 512 5376"
    "512 640 6720"
    "512 896 9408"
    "544 1280 14280"
    "544 800 8925"
    "544 960 10710"
    "640 640 8400"
    "736 1280 19320"
    "576 1024 12096"
)

for((i=0; i<${#RESOLUTIONS[@]}; i++))
do
    RESOLUTION=(`echo ${RESOLUTIONS[i]}`)
    H=${RESOLUTION[0]}
    W=${RESOLUTION[1]}

    python export_onnx.py \
    --output-name ${MODEL_NAME}_${SUFFIX}${H}x${W}.onnx \
    -n yolox-${TYPE} \
    -f ${TYPE}.py \
    -c ${MODEL_PATH} \
    -s "(${H}, ${W})"

    sng4onnx \
    --input_onnx_file_path ${MODEL_NAME}_${SUFFIX}${H}x${W}.onnx \
    --output_onnx_file_path ${MODEL_NAME}_${SUFFIX}${H}x${W}.onnx
done

python export_onnx.py \
--output-name ${MODEL_NAME}_1x3xHxW.onnx \
-n yolox-${TYPE} \
-f ${TYPE}.py \
-c ${MODEL_PATH} \
--dynamic

rm ${MODEL_NAME}_1x3xHxW.onnx