#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1wyM4AbwFRITCcHl-FepQ6yUO2jEIKRbN" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1wyM4AbwFRITCcHl-FepQ6yUO2jEIKRbN" -o mobilenetv2_fsd2018_41cls_integer_quant.tflite

echo Download finished.
