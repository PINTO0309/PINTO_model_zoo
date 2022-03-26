#!/bin/bash

fileid="1mxm4XAhZY8QKsye2-irSB4MgYmjKVXdC"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o lsmod_256_weight_quant.tflite
rm cookie

echo Download finished.
