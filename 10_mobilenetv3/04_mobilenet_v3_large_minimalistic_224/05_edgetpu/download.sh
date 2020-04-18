#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1ZxEygOWPD0y2_QaFb3X8l5d7W44W0S_Z" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1ZxEygOWPD0y2_QaFb3X8l5d7W44W0S_Z" -o mobilenet_v3_large_minimalistic_224_dm10_full_integer_quant_edgetpu.tflite

echo Download finished.
