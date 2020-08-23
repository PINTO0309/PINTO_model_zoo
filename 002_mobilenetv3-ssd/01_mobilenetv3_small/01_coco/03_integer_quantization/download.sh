#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1h23k090mcd1sUj_C2w679JM-Ixbza-YI" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1h23k090mcd1sUj_C2w679JM-Ixbza-YI" -o ssd_mobilenet_v3_small_coco_integer_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1ejDQyf6M-i3PPGoSM4SjHHTA7K-Uhc-O" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1ejDQyf6M-i3PPGoSM4SjHHTA7K-Uhc-O" -o ssd_mobilenet_v3_small_coco_integer_quant_postprocess.tflite

echo Download finished.
