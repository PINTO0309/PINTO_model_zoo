#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1xny0Fb_8Qv2_n0xfIVqtSqPuWVndXsXS" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1xny0Fb_8Qv2_n0xfIVqtSqPuWVndXsXS" -o monodepth2_colormap_depth_float16_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1t498UjHWXTsDNBxloK8edc0bIimDuPKm" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1t498UjHWXTsDNBxloK8edc0bIimDuPKm" -o monodepth2_colormap_only_float16_quant.tflite

echo Download finished.
