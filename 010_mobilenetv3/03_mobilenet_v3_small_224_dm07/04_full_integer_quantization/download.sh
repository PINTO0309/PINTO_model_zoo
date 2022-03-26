#!/bin/bash

fileid="1x2hgUiyTlCSWx8Isy8kvL3BFpZtc2Uaf"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v3_small_224_dm07_full_integer_quant.tflite

echo Download finished.
