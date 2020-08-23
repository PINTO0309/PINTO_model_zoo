#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1UeCXsjP-5vZN_E-7nfaSzo0zue1rD--S" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1UeCXsjP-5vZN_E-7nfaSzo0zue1rD--S" -o monodepth2_colormap_depth_integer_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1_Qc5GpVJk5JMmOGKBzDpMlej57Hxc5pi" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1_Qc5GpVJk5JMmOGKBzDpMlej57Hxc5pi" -o monodepth2_colormap_only_integer_quant.tflite

echo Download finished.
