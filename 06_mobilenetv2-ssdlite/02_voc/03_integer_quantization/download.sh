#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1GxUBbu_zRUQLY4gaerqifCV0AID3ViIU" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1GxUBbu_zRUQLY4gaerqifCV0AID3ViIU" -o ssdlite_mobilenet_v2_voc_300_integer_quant_with_postprocess.tflite

echo Download finished.
