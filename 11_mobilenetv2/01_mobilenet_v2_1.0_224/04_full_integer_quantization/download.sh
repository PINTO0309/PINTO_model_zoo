#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=11Ask0EUMpU83RYV_YpVgALjHYTDjikmD" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=11Ask0EUMpU83RYV_YpVgALjHYTDjikmD" -o mobilenet_v2_224_dm10_full_integer_quant.tflite

echo Download finished.
