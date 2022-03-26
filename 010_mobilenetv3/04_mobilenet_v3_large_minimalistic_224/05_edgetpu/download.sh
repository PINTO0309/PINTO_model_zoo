#!/bin/bash

fileid="1ZxEygOWPD0y2_QaFb3X8l5d7W44W0S_Z"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v3_large_minimalistic_224_dm10_full_integer_quant_edgetpu.tflite

echo Download finished.
