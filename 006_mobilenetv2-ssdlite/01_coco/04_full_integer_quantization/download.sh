#!/bin/bash

fileid="1F9rmhY5FeHS3po7HHTrF9ZBgovzg32DS"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ssdlite_mobilenet_v2_coco_300_full_integer_quant.tflite

echo Download finished.
