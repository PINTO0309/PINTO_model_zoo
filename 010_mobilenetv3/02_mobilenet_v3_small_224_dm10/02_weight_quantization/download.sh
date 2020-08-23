#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1IA4jONUoPhpIpxgCRXE0t8lB449ayrAs" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1IA4jONUoPhpIpxgCRXE0t8lB449ayrAs" -o mobilenet_v3_small_224_dm10_weight_quant.tflite

echo Download finished.
