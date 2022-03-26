#!/bin/bash

fileid="17Dzs-3F7ChGjlZa-0SB5-BhTLB8BICfA"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_lite_voc_256_weight_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1IBNTF_GjmkmFciMitsUlLG05hAFUN6NX" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1IBNTF_GjmkmFciMitsUlLG05hAFUN6NX" -o yolov3_lite_voc_320_weight_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1nK4sOCEITXxraZOQHy0Pma33l9KAgMLP" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1nK4sOCEITXxraZOQHy0Pma33l9KAgMLP" -o yolov3_lite_voc_416_weight_quant.tflite

echo Download finished.
