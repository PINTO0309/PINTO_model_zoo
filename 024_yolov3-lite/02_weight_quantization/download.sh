#!/bin/bash
# yolov3_lite_voc_256_weight_quant.tflite
fileid="17Dzs-3F7ChGjlZa-0SB5-BhTLB8BICfA"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_lite_voc_256_weight_quant.tflite
rm cookie

# yolov3_lite_voc_320_weight_quant.tflite
fileid="1IBNTF_GjmkmFciMitsUlLG05hAFUN6NX"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_lite_voc_320_weight_quant.tflite
rm cookie

# yolov3_lite_voc_416_weight_quant.tflite
fileid="1nK4sOCEITXxraZOQHy0Pma33l9KAgMLP"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_lite_voc_416_weight_quant.tflite
rm cookie

echo Download finished.
