#!/bin/bash

fileid="1Bai5fQdoX23kjCjsWP_sH5ygM_f69Jh6"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v3_large_224_dm10_full_integer_quant.tflite

echo Download finished.
