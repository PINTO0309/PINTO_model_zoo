#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1JfrfLfEn61TLUNW0CkRNMeLCHk_qSMXj" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1JfrfLfEn61TLUNW0CkRNMeLCHk_qSMXj" -o monodepth2_colormap_depth_float16_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Oqkoz-5u5Ltedc-bq0r-ELa6ahhPgmLG" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Oqkoz-5u5Ltedc-bq0r-ELa6ahhPgmLG" -o monodepth2_colormap_only_float16_quant.tflite

echo Download finished.
