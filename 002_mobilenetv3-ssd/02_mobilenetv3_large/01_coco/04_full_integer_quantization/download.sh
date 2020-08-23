#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1JDeHbaBk71JkHxkuyhvFjFEJn4CDQ4vr" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1JDeHbaBk71JkHxkuyhvFjFEJn4CDQ4vr" -o ssd_mobilenet_v3_large_coco_full_integer_quant.tflite

echo Download finished.
