#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1gIBuxbRgbQF2vS7hfc4y1gXwlthjZDIp" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1gIBuxbRgbQF2vS7hfc4y1gXwlthjZDIp" -o mobilenet_v3_large_224_dm07_weight_quant.tflite

echo Download finished.
