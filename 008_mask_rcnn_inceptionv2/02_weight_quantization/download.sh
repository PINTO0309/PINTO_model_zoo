#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1nE53eJt5mBOmzinGxHCR5eIcbghR80Ke" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1nE53eJt5mBOmzinGxHCR5eIcbghR80Ke" -o mask_rcnn_inception_v2_coco_256_weight_quant.tflite

echo Download finished.
