#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1UXpIKutJ0TQI25rjezuBr5rEt2aTo43e" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1UXpIKutJ0TQI25rjezuBr5rEt2aTo43e" -o mobilenet_v3_minimalistic_224_dm10_full_integer_quant_edgetpu.tflite

echo Download finished.
