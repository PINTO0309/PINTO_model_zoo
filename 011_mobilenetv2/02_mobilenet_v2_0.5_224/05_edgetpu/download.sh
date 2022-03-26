#!/bin/bash

fileid="1_KaBonDMJqBeYrrR9TzB7vfT6YyCfJtk"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v2_224_dm05_full_integer_quant_edgetpu.tflite

echo Download finished.
