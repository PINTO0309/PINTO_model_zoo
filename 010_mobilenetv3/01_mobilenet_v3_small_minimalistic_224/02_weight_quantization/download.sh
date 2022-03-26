#!/bin/bash

fileid="1wRYzar-NGGbYs8ijh4liZWirpuriE-ts"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v3_small_minimalistic_224_dm10_weight_quant.tflite

echo Download finished.
