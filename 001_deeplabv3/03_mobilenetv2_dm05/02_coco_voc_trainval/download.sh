#!/bin/bash

fileid="1wYPYH8Y8fV1BhckIXmrvBDXjyBrdkleH"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o deeplabv3_mnv2_dm05_pascal_trainval_weight_quant.tflite

echo Download finished.
