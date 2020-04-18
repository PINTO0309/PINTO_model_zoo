#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1zfaVokQjEwKACfa81kdfBoYN8o5DOyjo" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1zfaVokQjEwKACfa81kdfBoYN8o5DOyjo" -o mobilenet_v3_large_minimalistic_224_dm10_full_integer_quant.tflite

echo Download finished.
