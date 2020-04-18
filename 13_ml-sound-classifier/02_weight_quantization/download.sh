#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=10acElPx4Y_l319VOhVLzqy554Y4T7sWf" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=10acElPx4Y_l319VOhVLzqy554Y4T7sWf" -o mobilenetv2_fsd2018_41cls_weight_quant.tflite

echo Download finished.
