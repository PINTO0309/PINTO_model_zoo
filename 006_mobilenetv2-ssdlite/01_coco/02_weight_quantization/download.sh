#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=19J6eaYJw1UTph2SwwHjxLGh1qWZORFNr" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=19J6eaYJw1UTph2SwwHjxLGh1qWZORFNr" -o ssdlite_mobilenet_v2_coco_300_weight_quant.tflite

echo Download finished.
