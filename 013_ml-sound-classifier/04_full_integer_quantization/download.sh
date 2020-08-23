#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=18MROBXweoXl7mWVSkq80jSkfiL6dWd6O" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=18MROBXweoXl7mWVSkq80jSkfiL6dWd6O" -o mobilenetv2_fsd2018_41cls_full_integer_quant.tflite

echo Download finished.
