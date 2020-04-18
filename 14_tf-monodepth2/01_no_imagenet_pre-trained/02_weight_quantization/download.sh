#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1n_7320QwiBhgt4mkpS29XphwateLPLXF" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1n_7320QwiBhgt4mkpS29XphwateLPLXF" -o monodepth2_colormap_depth_weight_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Twla-dlOZ2s8vCMTHlDK4K5zipnAErHx" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Twla-dlOZ2s8vCMTHlDK4K5zipnAErHx" -o monodepth2_colormap_only_weight_quant.tflite

echo Download finished.
