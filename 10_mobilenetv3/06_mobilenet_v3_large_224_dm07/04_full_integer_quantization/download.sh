#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Scu-3Kdz36iIlaBvqJ3sOTLCvzxM71RI" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Scu-3Kdz36iIlaBvqJ3sOTLCvzxM71RI" -o mobilenet_v3_large_224_dm07_full_integer_quant.tflite

echo Download finished.
