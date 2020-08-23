#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1yVLzYEKQO9332hznm9z1o82X3DiOTZYd" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1yVLzYEKQO9332hznm9z1o82X3DiOTZYd" -o mobilenet_v3_large_minimalistic_224_dm10_integer_quant.tflite

echo Download finished.
