#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1_KaBonDMJqBeYrrR9TzB7vfT6YyCfJtk" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1_KaBonDMJqBeYrrR9TzB7vfT6YyCfJtk" -o mobilenet_v2_224_dm05_full_integer_quant_edgetpu.tflite

echo Download finished.
