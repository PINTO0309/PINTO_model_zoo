#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1E-19_wtOg3El6cuXiaB1iy8Dvvs80NOo" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1E-19_wtOg3El6cuXiaB1iy8Dvvs80NOo" -o yolov3_nano_voc_256_float16_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Xmu4i99ZNGCWCpVmOhcRvxRQDMYDu6h0" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Xmu4i99ZNGCWCpVmOhcRvxRQDMYDu6h0" -o yolov3_nano_voc_320_float16_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1tqvFG72dkdrcpKEIFv-ipoojmpmWfvzm" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1tqvFG72dkdrcpKEIFv-ipoojmpmWfvzm" -o yolov3_nano_voc_416_float16_quant.tflite

echo Download finished.
