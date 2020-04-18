#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1F9rmhY5FeHS3po7HHTrF9ZBgovzg32DS" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1F9rmhY5FeHS3po7HHTrF9ZBgovzg32DS" -o ssdlite_mobilenet_v2_coco_300_full_integer_quant.tflite

echo Download finished.
