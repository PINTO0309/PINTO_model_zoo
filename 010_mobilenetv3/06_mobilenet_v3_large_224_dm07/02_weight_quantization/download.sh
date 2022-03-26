#!/bin/bash

fileid="1gIBuxbRgbQF2vS7hfc4y1gXwlthjZDIp"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v3_large_224_dm07_.tflite

echo Download finished.
