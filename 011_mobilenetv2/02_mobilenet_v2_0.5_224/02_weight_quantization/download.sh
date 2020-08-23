#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Iqzz44OikPDKBdzJ-GLZwWNejw4QufiW" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Iqzz44OikPDKBdzJ-GLZwWNejw4QufiW" -o mobilenet_v2_224_dm05_weight_quant.tflite

echo Download finished.
