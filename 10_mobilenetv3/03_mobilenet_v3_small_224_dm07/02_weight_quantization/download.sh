#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1i7K4WHEqIwG53pvRIpUyhtndF4Y1jX9l" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1i7K4WHEqIwG53pvRIpUyhtndF4Y1jX9l" -o mobilenet_v3_small_224_dm07_weight_quant.tflite

echo Download finished.
