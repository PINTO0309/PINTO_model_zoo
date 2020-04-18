#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Zw8TgVoidM5TJ8T19kZ0ny-Xp2Cf515o" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Zw8TgVoidM5TJ8T19kZ0ny-Xp2Cf515o" -o mobilenet_v3_small_224_dm10_full_integer_quant.tflite

echo Download finished.
