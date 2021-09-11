MODEL=fast_depth

H=128;W=160
python3 -m onnxsim ${MODEL}_${H}x${W}.onnx ${MODEL}_${H}x${W}_opt.onnx
H=224;W=224
python3 -m onnxsim ${MODEL}_${H}x${W}.onnx ${MODEL}_${H}x${W}_opt.onnx
H=256;W=256
python3 -m onnxsim ${MODEL}_${H}x${W}.onnx ${MODEL}_${H}x${W}_opt.onnx
H=256;W=320
python3 -m onnxsim ${MODEL}_${H}x${W}.onnx ${MODEL}_${H}x${W}_opt.onnx
H=320;W=320
python3 -m onnxsim ${MODEL}_${H}x${W}.onnx ${MODEL}_${H}x${W}_opt.onnx
H=480;W=640
python3 -m onnxsim ${MODEL}_${H}x${W}.onnx ${MODEL}_${H}x${W}_opt.onnx
H=512;W=512
python3 -m onnxsim ${MODEL}_${H}x${W}.onnx ${MODEL}_${H}x${W}_opt.onnx
H=768;W=1280
python3 -m onnxsim ${MODEL}_${H}x${W}.onnx ${MODEL}_${H}x${W}_opt.onnx

H=128;W=160
mv ${MODEL}_${H}x${W}_opt.onnx ${MODEL}_${H}x${W}.onnx
H=224;W=224
mv ${MODEL}_${H}x${W}_opt.onnx ${MODEL}_${H}x${W}.onnx
H=256;W=256
mv ${MODEL}_${H}x${W}_opt.onnx ${MODEL}_${H}x${W}.onnx
H=256;W=320
mv ${MODEL}_${H}x${W}_opt.onnx ${MODEL}_${H}x${W}.onnx
H=320;W=320
mv ${MODEL}_${H}x${W}_opt.onnx ${MODEL}_${H}x${W}.onnx
H=480;W=640
mv ${MODEL}_${H}x${W}_opt.onnx ${MODEL}_${H}x${W}.onnx
H=512;W=512
mv ${MODEL}_${H}x${W}_opt.onnx ${MODEL}_${H}x${W}.onnx
H=768;W=1280
mv ${MODEL}_${H}x${W}_opt.onnx ${MODEL}_${H}x${W}.onnx


xhost +local: && \
docker run --gpus all -it --rm \
-v `pwd`:/home/user/workdir \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
pinto0309/openvino2tensorflow:latest

cd workdir

MODEL=fast_depth
H=128;W=160
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP32 \
--output_dir ${H}x${W}/openvino/FP32
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP16 \
--output_dir ${H}x${W}/openvino/FP16
mkdir -p ${H}x${W}/openvino/myriad
${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/myriad_compile \
-m ${H}x${W}/openvino/FP16/${MODEL}_${H}x${W}.xml \
-ip U8 \
-VPU_NUMBER_OF_SHAVES 4 \
-VPU_NUMBER_OF_CMX_SLICES 4 \
-o ${H}x${W}/openvino/myriad/${MODEL}_${H}x${W}.blob

H=224;W=224
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP32 \
--output_dir ${H}x${W}/openvino/FP32
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP16 \
--output_dir ${H}x${W}/openvino/FP16
mkdir -p ${H}x${W}/openvino/myriad
${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/myriad_compile \
-m ${H}x${W}/openvino/FP16/${MODEL}_${H}x${W}.xml \
-ip U8 \
-VPU_NUMBER_OF_SHAVES 4 \
-VPU_NUMBER_OF_CMX_SLICES 4 \
-o ${H}x${W}/openvino/myriad/${MODEL}_${H}x${W}.blob

H=256;W=256
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP32 \
--output_dir ${H}x${W}/openvino/FP32
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP16 \
--output_dir ${H}x${W}/openvino/FP16
mkdir -p ${H}x${W}/openvino/myriad
${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/myriad_compile \
-m ${H}x${W}/openvino/FP16/${MODEL}_${H}x${W}.xml \
-ip U8 \
-VPU_NUMBER_OF_SHAVES 4 \
-VPU_NUMBER_OF_CMX_SLICES 4 \
-o ${H}x${W}/openvino/myriad/${MODEL}_${H}x${W}.blob

H=256;W=320
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP32 \
--output_dir ${H}x${W}/openvino/FP32
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP16 \
--output_dir ${H}x${W}/openvino/FP16
mkdir -p ${H}x${W}/openvino/myriad
${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/myriad_compile \
-m ${H}x${W}/openvino/FP16/${MODEL}_${H}x${W}.xml \
-ip U8 \
-VPU_NUMBER_OF_SHAVES 4 \
-VPU_NUMBER_OF_CMX_SLICES 4 \
-o ${H}x${W}/openvino/myriad/${MODEL}_${H}x${W}.blob

H=320;W=320
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP32 \
--output_dir ${H}x${W}/openvino/FP32
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP16 \
--output_dir ${H}x${W}/openvino/FP16
mkdir -p ${H}x${W}/openvino/myriad
${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/myriad_compile \
-m ${H}x${W}/openvino/FP16/${MODEL}_${H}x${W}.xml \
-ip U8 \
-VPU_NUMBER_OF_SHAVES 4 \
-VPU_NUMBER_OF_CMX_SLICES 4 \
-o ${H}x${W}/openvino/myriad/${MODEL}_${H}x${W}.blob

H=480;W=640
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP32 \
--output_dir ${H}x${W}/openvino/FP32
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP16 \
--output_dir ${H}x${W}/openvino/FP16
mkdir -p ${H}x${W}/openvino/myriad
${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/myriad_compile \
-m ${H}x${W}/openvino/FP16/${MODEL}_${H}x${W}.xml \
-ip U8 \
-VPU_NUMBER_OF_SHAVES 4 \
-VPU_NUMBER_OF_CMX_SLICES 4 \
-o ${H}x${W}/openvino/myriad/${MODEL}_${H}x${W}.blob

H=512;W=512
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP32 \
--output_dir ${H}x${W}/openvino/FP32
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP16 \
--output_dir ${H}x${W}/openvino/FP16
mkdir -p ${H}x${W}/openvino/myriad
${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/myriad_compile \
-m ${H}x${W}/openvino/FP16/${MODEL}_${H}x${W}.xml \
-ip U8 \
-VPU_NUMBER_OF_SHAVES 4 \
-VPU_NUMBER_OF_CMX_SLICES 4 \
-o ${H}x${W}/openvino/myriad/${MODEL}_${H}x${W}.blob

H=768;W=1280
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP32 \
--output_dir ${H}x${W}/openvino/FP32
$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer/mo.py \
--input_model ${MODEL}_${H}x${W}.onnx \
--data_type FP16 \
--output_dir ${H}x${W}/openvino/FP16
mkdir -p ${H}x${W}/openvino/myriad
${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/myriad_compile \
-m ${H}x${W}/openvino/FP16/${MODEL}_${H}x${W}.xml \
-ip U8 \
-VPU_NUMBER_OF_SHAVES 4 \
-VPU_NUMBER_OF_CMX_SLICES 4 \
-o ${H}x${W}/openvino/myriad/${MODEL}_${H}x${W}.blob

MODEL=fast_depth

H=128;W=160
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--output_no_quant_float32_tflite \
--output_weight_quant_tflite \
--output_float16_quant_tflite \
--output_integer_quant_tflite \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_tfjs \
--output_coreml \
--output_tftrt
mv saved_model saved_model_${H}x${W}
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_edgetpu
mv saved_model/model_full_integer_quant.tflite saved_model_${H}x${W}/model_full_integer_quant.tflite
mv saved_model/model_full_integer_quant_edgetpu.tflite saved_model_${H}x${W}/model_full_integer_quant_edgetpu.tflite
rm -rf saved_model
mv ${H}x${W}/openvino saved_model_${H}x${W}/openvino


H=224;W=224
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--output_no_quant_float32_tflite \
--output_weight_quant_tflite \
--output_float16_quant_tflite \
--output_integer_quant_tflite \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_tfjs \
--output_coreml \
--output_tftrt
mv saved_model saved_model_${H}x${W}
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_edgetpu
mv saved_model/model_full_integer_quant.tflite saved_model_${H}x${W}/model_full_integer_quant.tflite
mv saved_model/model_full_integer_quant_edgetpu.tflite saved_model_${H}x${W}/model_full_integer_quant_edgetpu.tflite
rm -rf saved_model
mv ${H}x${W}/openvino saved_model_${H}x${W}/openvino


H=256;W=256
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--output_no_quant_float32_tflite \
--output_weight_quant_tflite \
--output_float16_quant_tflite \
--output_integer_quant_tflite \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_tfjs \
--output_coreml \
--output_tftrt
mv saved_model saved_model_${H}x${W}
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_edgetpu
mv saved_model/model_full_integer_quant.tflite saved_model_${H}x${W}/model_full_integer_quant.tflite
mv saved_model/model_full_integer_quant_edgetpu.tflite saved_model_${H}x${W}/model_full_integer_quant_edgetpu.tflite
rm -rf saved_model
mv ${H}x${W}/openvino saved_model_${H}x${W}/openvino


H=256;W=320
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--output_no_quant_float32_tflite \
--output_weight_quant_tflite \
--output_float16_quant_tflite \
--output_integer_quant_tflite \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_tfjs \
--output_coreml \
--output_tftrt
mv saved_model saved_model_${H}x${W}
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_edgetpu
mv saved_model/model_full_integer_quant.tflite saved_model_${H}x${W}/model_full_integer_quant.tflite
mv saved_model/model_full_integer_quant_edgetpu.tflite saved_model_${H}x${W}/model_full_integer_quant_edgetpu.tflite
rm -rf saved_model
mv ${H}x${W}/openvino saved_model_${H}x${W}/openvino


H=320;W=320
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--output_no_quant_float32_tflite \
--output_weight_quant_tflite \
--output_float16_quant_tflite \
--output_integer_quant_tflite \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_tfjs \
--output_coreml \
--output_tftrt
mv saved_model saved_model_${H}x${W}
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_edgetpu
mv saved_model/model_full_integer_quant.tflite saved_model_${H}x${W}/model_full_integer_quant.tflite
mv saved_model/model_full_integer_quant_edgetpu.tflite saved_model_${H}x${W}/model_full_integer_quant_edgetpu.tflite
rm -rf saved_model
mv ${H}x${W}/openvino saved_model_${H}x${W}/openvino


H=480;W=640
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--output_no_quant_float32_tflite \
--output_weight_quant_tflite \
--output_float16_quant_tflite \
--output_integer_quant_tflite \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_tfjs \
--output_coreml \
--output_tftrt
mv saved_model saved_model_${H}x${W}
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_edgetpu
mv saved_model/model_full_integer_quant.tflite saved_model_${H}x${W}/model_full_integer_quant.tflite
mv saved_model/model_full_integer_quant_edgetpu.tflite saved_model_${H}x${W}/model_full_integer_quant_edgetpu.tflite
rm -rf saved_model
mv ${H}x${W}/openvino saved_model_${H}x${W}/openvino


H=512;W=512
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--output_no_quant_float32_tflite \
--output_weight_quant_tflite \
--output_float16_quant_tflite \
--output_integer_quant_tflite \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_tfjs \
--output_coreml \
--output_tftrt
mv saved_model saved_model_${H}x${W}
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_edgetpu
mv saved_model/model_full_integer_quant.tflite saved_model_${H}x${W}/model_full_integer_quant.tflite
mv saved_model/model_full_integer_quant_edgetpu.tflite saved_model_${H}x${W}/model_full_integer_quant_edgetpu.tflite
rm -rf saved_model
mv ${H}x${W}/openvino saved_model_${H}x${W}/openvino


H=768;W=1280
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--output_no_quant_float32_tflite \
--output_weight_quant_tflite \
--output_float16_quant_tflite \
--output_integer_quant_tflite \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_tfjs \
--output_coreml \
--output_tftrt
mv saved_model saved_model_${H}x${W}
openvino2tensorflow \
--model_path ${H}x${W}/openvino/FP32/${MODEL}_${H}x${W}.xml \
--output_saved_model \
--output_pb \
--string_formulas_for_normalization 'data / 255' \
--output_integer_quant_type 'uint8' \
--output_edgetpu
mv saved_model/model_full_integer_quant.tflite saved_model_${H}x${W}/model_full_integer_quant.tflite
mv saved_model/model_full_integer_quant_edgetpu.tflite saved_model_${H}x${W}/model_full_integer_quant_edgetpu.tflite
rm -rf saved_model
mv ${H}x${W}/openvino saved_model_${H}x${W}/openvino

MODEL=fast_depth
H=128;W=160
mv ${MODEL}_${H}x${W}.onnx saved_model_${H}x${W}/${MODEL}_${H}x${W}.onnx
H=224;W=224
mv ${MODEL}_${H}x${W}.onnx saved_model_${H}x${W}/${MODEL}_${H}x${W}.onnx
H=256;W=256
mv ${MODEL}_${H}x${W}.onnx saved_model_${H}x${W}/${MODEL}_${H}x${W}.onnx
H=256;W=320
mv ${MODEL}_${H}x${W}.onnx saved_model_${H}x${W}/${MODEL}_${H}x${W}.onnx
H=320;W=320
mv ${MODEL}_${H}x${W}.onnx saved_model_${H}x${W}/${MODEL}_${H}x${W}.onnx
H=480;W=640
mv ${MODEL}_${H}x${W}.onnx saved_model_${H}x${W}/${MODEL}_${H}x${W}.onnx
H=512;W=512
mv ${MODEL}_${H}x${W}.onnx saved_model_${H}x${W}/${MODEL}_${H}x${W}.onnx
H=768;W=1280
mv ${MODEL}_${H}x${W}.onnx saved_model_${H}x${W}/${MODEL}_${H}x${W}.onnx
