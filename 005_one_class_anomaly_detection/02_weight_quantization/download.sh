#!/bin/bash

fileid="1JSFSfgwI1zP9JanSTbyY4NgRkjPVlSvU"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o weights_weight_quant.tflite

echo Download finished.
