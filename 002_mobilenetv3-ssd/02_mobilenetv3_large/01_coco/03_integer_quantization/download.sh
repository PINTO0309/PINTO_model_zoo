#!/bin/bash

fileid="1Qphn11jnRJjEIniDG7uFUIpyvmWIKdoo"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ssd_mobilenet_v3_large_coco_integer_quant.tflite

fileid="13Za2i57aBJE4Bbp6SvGRcw3lHNgWf3J9"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ssd_mobilenet_v3_large_coco_integer_quant_postprocess.tflite

echo Download finished.
