#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1x2hgUiyTlCSWx8Isy8kvL3BFpZtc2Uaf" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1x2hgUiyTlCSWx8Isy8kvL3BFpZtc2Uaf" -o mobilenet_v3_small_224_dm07_full_integer_quant.tflite

echo Download finished.
