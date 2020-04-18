#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1apsOAiHolxihqNkacLhxXfMlRIPi8kgg" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1apsOAiHolxihqNkacLhxXfMlRIPi8kgg" -o mobilenet_v3_small_224_dm10_integer_quant.tflite

echo Download finished.
