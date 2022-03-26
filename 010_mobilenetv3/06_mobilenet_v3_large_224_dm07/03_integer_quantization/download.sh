#!/bin/bash

fileid="1MeUGlL6xMbhB_pc1be5lKYDkVMnEgQ1g"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v3_large_224_dm07_integer_quant.tflite

echo Download finished.
