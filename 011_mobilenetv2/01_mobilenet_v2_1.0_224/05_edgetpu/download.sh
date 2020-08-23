#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1PCj1ffojhfu9rvx4FwRfldr0vYsYViS0" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1PCj1ffojhfu9rvx4FwRfldr0vYsYViS0" -o mobilenet_v2_224_dm10_full_integer_quant_edgetpu.tflite

echo Download finished.
