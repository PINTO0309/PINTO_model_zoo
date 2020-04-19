#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1vBUtdzwcUw0e9D-sUP5I1A1TEwsrCnkX" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1vBUtdzwcUw0e9D-sUP5I1A1TEwsrCnkX" -o yolov3_nano_voc_256_weight_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=18Ygr4HwdN51_5G2Nz4Wt4zf3j5Y35gcr" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=18Ygr4HwdN51_5G2Nz4Wt4zf3j5Y35gcr" -o yolov3_nano_voc_320_weight_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=15aukwxJHfhv5fH3jx3wc4GUB4Yq2e3hV" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=15aukwxJHfhv5fH3jx3wc4GUB4Yq2e3hV" -o yolov3_nano_voc_416_weight_quant.tflite

echo Download finished.