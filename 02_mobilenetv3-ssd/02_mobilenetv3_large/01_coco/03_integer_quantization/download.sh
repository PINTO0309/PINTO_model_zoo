#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Qphn11jnRJjEIniDG7uFUIpyvmWIKdoo" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Qphn11jnRJjEIniDG7uFUIpyvmWIKdoo" -o ssd_mobilenet_v3_large_coco_integer_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=13Za2i57aBJE4Bbp6SvGRcw3lHNgWf3J9" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=13Za2i57aBJE4Bbp6SvGRcw3lHNgWf3J9" -o ssd_mobilenet_v3_large_coco_integer_quant_postprocess.tflite

echo Download finished.
