#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1JVgod2VWYhvXb9B0-gIiOefxpS_xcVPy" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1JVgod2VWYhvXb9B0-gIiOefxpS_xcVPy" -o mask_rcnn_inception_v2_coco_256_float16_quant.tflite

echo Download finished.
