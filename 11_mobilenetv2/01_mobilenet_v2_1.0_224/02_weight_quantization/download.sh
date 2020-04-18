#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=16vS_gGJ64vxHOIGojho3mqTyf49B6WUb" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=16vS_gGJ64vxHOIGojho3mqTyf49B6WUb" -o mobilenet_v2_224_dm10_weight_quant.tflite

echo Download finished.
