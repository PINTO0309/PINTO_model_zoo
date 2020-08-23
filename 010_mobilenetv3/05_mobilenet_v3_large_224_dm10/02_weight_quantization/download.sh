#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=197h2mgWGwEIHanWok6B9QtobKBVkjV0I" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=197h2mgWGwEIHanWok6B9QtobKBVkjV0I" -o mobilenet_v3_large_224_dm10_weight_quant.tflite

echo Download finished.
