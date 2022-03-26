#!/bin/bash

fileid="1vBUtdzwcUw0e9D-sUP5I1A1TEwsrCnkX"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_nano_voc_256_weight_quant.tflite

fileid="18Ygr4HwdN51_5G2Nz4Wt4zf3j5Y35gcr"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_nano_voc_320_weight_quant.tflite

fileid="15aukwxJHfhv5fH3jx3wc4GUB4Yq2e3hV"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_nano_voc_416_weight_quant.tflite

echo Download finished.