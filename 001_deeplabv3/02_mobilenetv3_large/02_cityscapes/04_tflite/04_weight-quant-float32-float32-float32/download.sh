#!/bin/bash

fileid="1wwO8hjoilIDDPEwc1acXN7sEINTl3Rgi"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o deeplab_mnv3_large_weight_quant_257.tflite

fileid="1OSKU0ssLzsZYqdMNezAEgSNB0EAdnajz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o deeplab_mnv3_large_weight_quant_769.tflite

echo Download finished.
