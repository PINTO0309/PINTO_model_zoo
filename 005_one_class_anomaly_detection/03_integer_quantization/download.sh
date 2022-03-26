#!/bin/bash

fileid="1Cdmb_cb8IU2p6_xMWsW6SiQ1OgMSiOeJ"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o weights_integer_quant.tflite

echo Download finished.
