#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=11wCtdIQMZ06MuTQWj8HKNQT3_4DHdhle" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=11wCtdIQMZ06MuTQWj8HKNQT3_4DHdhle" -o mobilenet_v2_224_dm10_integer_quant.tflite

echo Download finished.
