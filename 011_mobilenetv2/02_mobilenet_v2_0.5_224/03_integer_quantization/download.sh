#!/bin/bash

fileid="1lspYx8gs63X0RxS9N9s4VPIKFNdRpahB"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v2_224_dm05_integer_quant.tflite

echo Download finished.
