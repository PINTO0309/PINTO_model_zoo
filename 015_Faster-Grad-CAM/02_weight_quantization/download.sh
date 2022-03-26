#!/bin/bash

fileid="1FB-wB5a_1FhHdQ1RD6OMwxTA1QjHjPXs"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o weights_weight_quant.tflite

echo Download finished.
