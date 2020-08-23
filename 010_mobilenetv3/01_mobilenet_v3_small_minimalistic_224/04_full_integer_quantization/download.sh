#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=19H9dRPcYUh5kfizEiKxN4JrolNOLq2dT" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=19H9dRPcYUh5kfizEiKxN4JrolNOLq2dT" -o mobilenet_v3_small_minimalistic_224_dm10_full_integer_quant.tflite

echo Download finished.
