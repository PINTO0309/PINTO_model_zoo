#!/bin/bash

RESOLUTIONS=(
    "_1 180 416"
    "_2 180 512"
    "_3 180 640"
    "_4 180 800"
    "_5 240 320"
    "_6 240 416"
    "_7 240 512"
    "_8 240 640"
    "_9 240 800"
    "_10 240 960"
    "_11 288 480"
    "_12 288 512"
    "_13 288 640"
    "_14 288 800"
    "_15 288 960"
    "_16 288 1280"
    "_17 320 320"
    "_18 360 480"
    "_19 360 512"
    "_20 360 640"
    "_21 360 800"
    "_22 360 960"
    "_23 360 1280"
    "_24 416 416"
    "_25 480 640"
    "_26 480 800"
    "_27 480 960"
    "_28 480 1280"
    "_29 512 512"
    "_30 540 800"
    "_31 540 960"
    "_32 540 1280"
    "_33 640 640"
    "_34 720 1280"
)

for((j=0; j<${#RESOLUTIONS[@]}; j++))
do
    RESOLUTION=(`echo ${RESOLUTIONS[j]}`)
    N=${RESOLUTION[0]}
    H=${RESOLUTION[1]}
    W=${RESOLUTION[2]}
    echo @@@@@@@@@@@@@@@@@ processing ${MODELNAME} ...

    saved_model_to_tflite \
    --saved_model_dir_path saved_model_${H}x${W} \
    --output_no_quant_float32_tflite \
    --model_output_dir_path saved_model_${H}x${W}

    python -m tf2onnx.convert \
    --opset 13 \
    --inputs-as-nchw input${N} \
    --tflite saved_model_${H}x${W}/model_float32.tflite \
    --output saved_model_${H}x${W}/mspfn_${H}x${W}.onnx

    sor4onnx \
    --input_onnx_file_path saved_model_${H}x${W}/mspfn_${H}x${W}.onnx \
    --old_new "input${N}" "input" \
    --mode inputs \
    --search_mode exact_match \
    --output_onnx_file_path saved_model_${H}x${W}/mspfn_${H}x${W}.onnx

    sor4onnx \
    --input_onnx_file_path saved_model_${H}x${W}/mspfn_${H}x${W}.onnx \
    --old_new "sub${N}" "output" \
    --mode outputs \
    --search_mode exact_match \
    --output_onnx_file_path saved_model_${H}x${W}/mspfn_${H}x${W}.onnx

    # onnx2tf -i saved_model_${H}x${W}/mspfn_${H}x${W}.onnx -n

    rm saved_model_${H}x${W}/model_float32.tflite
    mv saved_model/mspfn_${H}x${W}_float32.tflite saved_model_${H}x${W}
    mv saved_model/mspfn_${H}x${W}_float16.tflite saved_model_${H}x${W}

    rm -rf saved_model
done
