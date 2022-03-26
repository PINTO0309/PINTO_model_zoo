#!/bin/bash

fileid="1HRu3Ua2ra1LOgJFGkBDNFBRv5gT94m54"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o weights_full_integer_quant_edgetpu.tflite

echo Download finished.
