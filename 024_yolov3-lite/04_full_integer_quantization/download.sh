#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=19Vcthxad-odbcUF5zLF-Y12ADOXG_q0o" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=19Vcthxad-odbcUF5zLF-Y12ADOXG_q0o" -o yolov3_lite_voc_256_full_integer_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1z90PlRLtZolwqW-3I3_FV2HEljylOIAi" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1z90PlRLtZolwqW-3I3_FV2HEljylOIAi" -o yolov3_lite_voc_320_full_integer_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1QBiU5gGJwV5Jcb488RCqwgVNfvARhY_u" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1QBiU5gGJwV5Jcb488RCqwgVNfvARhY_u" -o yolov3_lite_voc_416_full_integer_quant.tflite

echo Download finished.
