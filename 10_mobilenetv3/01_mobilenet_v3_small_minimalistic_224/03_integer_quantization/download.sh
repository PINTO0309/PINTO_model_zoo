#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1NLBRZxETIjzSNUuVzS8DJHMXWnhrfyiC" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1NLBRZxETIjzSNUuVzS8DJHMXWnhrfyiC" -o mobilenet_v3_small_minimalistic_224_dm10_integer_quant.tflite

echo Download finished.
