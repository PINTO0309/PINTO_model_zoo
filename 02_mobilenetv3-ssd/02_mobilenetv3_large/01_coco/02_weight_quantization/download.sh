#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1MDcc8uHPMM7DjutW8V9DB3RHJkslLDQc" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1MDcc8uHPMM7DjutW8V9DB3RHJkslLDQc" -o ssd_mobilenet_v3_large_coco_weight_quant.tflite

echo Download finished.
