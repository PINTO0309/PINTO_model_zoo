#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1nqymGXEdd4XkuINHyLbVFxV996k94NS2" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1nqymGXEdd4XkuINHyLbVFxV996k94NS2" -o mobilenet_v3_small_224_dm07_integer_quant.tflite

echo Download finished.
