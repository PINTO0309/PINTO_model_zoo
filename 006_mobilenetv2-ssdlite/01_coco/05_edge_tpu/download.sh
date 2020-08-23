#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1_8bFUcATFt5apHpA4Ym9GyJNo2_dGz-M" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1_8bFUcATFt5apHpA4Ym9GyJNo2_dGz-M" -o ssdlite_mobilenet_v2_coco_300_full_integer_quant_edgetpu.tflite

echo Download finished.
