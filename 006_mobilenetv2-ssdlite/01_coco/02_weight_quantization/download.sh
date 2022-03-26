#!/bin/bash

fileid="19J6eaYJw1UTph2SwwHjxLGh1qWZORFNr"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ssdlite_mobilenet_v2_coco_300_weight_quant.tflite

echo Download finished.
