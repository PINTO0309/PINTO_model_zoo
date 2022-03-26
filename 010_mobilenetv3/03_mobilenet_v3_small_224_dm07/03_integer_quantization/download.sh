#!/bin/bash

fileid="1nqymGXEdd4XkuINHyLbVFxV996k94NS2"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v3_small_224_dm07_integer_quant.tflite

echo Download finished.
