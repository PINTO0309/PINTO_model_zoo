#!/bin/bash

fileid="1UXpIKutJ0TQI25rjezuBr5rEt2aTo43e"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v3_minimalistic_224_dm10_full_integer_quant_edgetpu.tflite

echo Download finished.
