#!/bin/bash

fileid="1vRWYDARVyRXNgCq5dzsGD9Wb3TgB_up2"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o efficientnet_b0_224_integer_quant.tflite

echo Download finished.
