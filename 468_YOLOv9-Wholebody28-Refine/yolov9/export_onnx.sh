TYPE=e
# RELU= or RELU=-relu
RELU=
RELUS=$(echo ${RELU} | sed 's/-/_/g')
MODEL_NAME=yolov9_${TYPE}_wholebody28
SUFFIX="0100_1x3x"
# best-t.pt
# best-t-relu.pt
# best-e.pt
# best-e-relu.pt
MODEL_PATH=best-${TYPE}${RELU}.pt #best-${TYPE}${RELU}.pt
OPSET=13 # default: 13, for onnxruntime-web: 11

RESOLUTIONS=(
    "128 160"
    "128 256"
    "192 320"
    "192 416"
    "192 640"
    "192 800"
    "256 320"
    "256 416"
    "256 448"
    "256 640"
    "256 800"
    "256 960"
    "288 1280"
    "288 480"
    "288 640"
    "288 800"
    "288 960"
    "320 320"
    "384 1280"
    "384 480"
    "384 640"
    "384 800"
    "384 960"
    "416 416"
    "480 1280"
    "480 640"
    "480 800"
    "480 960"
    "512 512"
    "512 640"
    "512 896"
    "544 1280"
    "544 800"
    "544 960"
    "640 640"
    "736 1280"
    "576 1024"
    "384 672"
)

python reparameterization${RELUS}.py \
--type ${TYPE} \
--cfg ./models/detect/gelan-${TYPE}${RELU}.yaml \
--weights ${MODEL_PATH} \
--save ${MODEL_NAME}${RELUS}.pt

for((i=0; i<${#RESOLUTIONS[@]}; i++))
do
    RESOLUTION=(`echo ${RESOLUTIONS[i]}`)
    H=${RESOLUTION[0]}
    W=${RESOLUTION[1]}

    python export.py \
    --data data/original.yaml \
    --weights ${MODEL_NAME}${RELUS}.pt \
    --imgsz ${H} ${W} \
    --batch-size 1 \
    --device cpu \
    --opset ${OPSET} \
    --include onnx

    mv ${MODEL_NAME}${RELUS}.onnx ${MODEL_NAME}${RELUS}_${SUFFIX}${H}x${W}.onnx

    sng4onnx \
    --input_onnx_file_path ${MODEL_NAME}${RELUS}_${SUFFIX}${H}x${W}.onnx \
    --output_onnx_file_path ${MODEL_NAME}${RELUS}_${SUFFIX}${H}x${W}.onnx

    onnxsim ${MODEL_NAME}${RELUS}_${SUFFIX}${H}x${W}.onnx ${MODEL_NAME}${RELUS}_${SUFFIX}${H}x${W}.onnx
    onnxsim ${MODEL_NAME}${RELUS}_${SUFFIX}${H}x${W}.onnx ${MODEL_NAME}${RELUS}_${SUFFIX}${H}x${W}.onnx
    onnxsim ${MODEL_NAME}${RELUS}_${SUFFIX}${H}x${W}.onnx ${MODEL_NAME}${RELUS}_${SUFFIX}${H}x${W}.onnx
done

python export.py \
--data data/original.yaml \
--weights ${MODEL_NAME}${RELUS}.pt \
--device cpu \
--opset ${OPSET} \
--include onnx \
--dynamic
mv ${MODEL_NAME}${RELUS}.onnx ${MODEL_NAME}${RELUS}_Nx3HxW.onnx
onnxsim ${MODEL_NAME}${RELUS}_Nx3HxW.onnx ${MODEL_NAME}${RELUS}_Nx3HxW.onnx
