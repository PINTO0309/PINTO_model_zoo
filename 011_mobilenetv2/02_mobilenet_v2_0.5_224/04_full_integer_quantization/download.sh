#!/bin/bash

fileid="1Hn-3nuEoQ5MOE4e_nFFQnEAmmn8V9t8f"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v2_224_dm05_full_integer_quant.tflite

echo Download finished.
