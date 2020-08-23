#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1DpdWUGbACcTrasPN7U-NkLAtl9IJphV3" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1DpdWUGbACcTrasPN7U-NkLAtl9IJphV3" -o yolov3_lite_voc_256_integer_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1pE8fyw7jU2HxIcbM-lX9p_8dMv03sMLc" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1pE8fyw7jU2HxIcbM-lX9p_8dMv03sMLc" -o yolov3_lite_voc_320_integer_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1lIHGlslrH3xHx1wBVEnQv1YH88mpVxkd" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1lIHGlslrH3xHx1wBVEnQv1YH88mpVxkd" -o yolov3_lite_voc_416_integer_quant.tflite

echo Download finished.
