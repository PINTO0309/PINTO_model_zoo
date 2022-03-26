#!/bin/bash

fileid="1GxUBbu_zRUQLY4gaerqifCV0AID3ViIU"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ssdlite_mobilenet_v2_voc_300_integer_quant_with_postprocess.tflite

echo Download finished.
