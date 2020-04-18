#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1lspYx8gs63X0RxS9N9s4VPIKFNdRpahB" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1lspYx8gs63X0RxS9N9s4VPIKFNdRpahB" -o mobilenet_v2_224_dm05_integer_quant.tflite

echo Download finished.
