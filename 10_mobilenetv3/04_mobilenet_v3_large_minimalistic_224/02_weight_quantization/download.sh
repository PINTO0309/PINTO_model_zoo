#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=12cApnDgEBHqsOluljW7KNIMYvoGoQ3Ua" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=12cApnDgEBHqsOluljW7KNIMYvoGoQ3Ua" -o mobilenet_v3_large_minimalistic_224_dm10_weight_quant.tflite

echo Download finished.
