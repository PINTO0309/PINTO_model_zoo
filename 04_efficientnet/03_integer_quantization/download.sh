#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1vRWYDARVyRXNgCq5dzsGD9Wb3TgB_up2" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1vRWYDARVyRXNgCq5dzsGD9Wb3TgB_up2" -o efficientnet_b0_224_integer_quant.tflite

echo Download finished.
