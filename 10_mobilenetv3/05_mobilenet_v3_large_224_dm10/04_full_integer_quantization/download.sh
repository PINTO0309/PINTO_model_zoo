#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Bai5fQdoX23kjCjsWP_sH5ygM_f69Jh6" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Bai5fQdoX23kjCjsWP_sH5ygM_f69Jh6" -o mobilenet_v3_large_224_dm10_full_integer_quant.tflite

echo Download finished.
