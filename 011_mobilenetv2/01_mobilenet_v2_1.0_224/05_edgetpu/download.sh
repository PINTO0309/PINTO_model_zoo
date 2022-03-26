#!/bin/bash

fileid="1PCj1ffojhfu9rvx4FwRfldr0vYsYViS0"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v2_224_dm10_full_integer_quant_edgetpu.tflite

echo Download finished.
