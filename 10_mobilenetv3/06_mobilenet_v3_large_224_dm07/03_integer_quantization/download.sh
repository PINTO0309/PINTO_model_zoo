#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1MeUGlL6xMbhB_pc1be5lKYDkVMnEgQ1g" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1MeUGlL6xMbhB_pc1be5lKYDkVMnEgQ1g" -o mobilenet_v3_large_224_dm07_integer_quant.tflite

echo Download finished.
