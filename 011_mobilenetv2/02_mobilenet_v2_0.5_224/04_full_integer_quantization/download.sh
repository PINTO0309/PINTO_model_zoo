#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Hn-3nuEoQ5MOE4e_nFFQnEAmmn8V9t8f" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Hn-3nuEoQ5MOE4e_nFFQnEAmmn8V9t8f" -o mobilenet_v2_224_dm05_full_integer_quant.tflite

echo Download finished.
