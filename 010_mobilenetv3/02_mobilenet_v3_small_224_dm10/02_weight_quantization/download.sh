#!/bin/bash

fileid="1IA4jONUoPhpIpxgCRXE0t8lB449ayrAs"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v3_small_224_dm10_weight_quant.tflite

echo Download finished.
