#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1wYPYH8Y8fV1BhckIXmrvBDXjyBrdkleH" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1wYPYH8Y8fV1BhckIXmrvBDXjyBrdkleH" -o deeplabv3_mnv2_dm05_pascal_trainval_weight_quant.tflite

echo Download finished.
