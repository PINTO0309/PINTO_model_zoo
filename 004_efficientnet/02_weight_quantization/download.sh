#!/bin/bash

fileid="1K0hKFxQicJfGMlu4kx0vuPeye6HddiSH"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o efficientnet_b0_224_weight_quant.tflite

echo Download finished.
