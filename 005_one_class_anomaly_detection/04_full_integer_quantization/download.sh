#!/bin/bash

fileid="1T9IlVn2PWX5bFg4lCe8wd6ZFHRx5XjzA"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o weights_full_integer_quant.tflite

echo Download finished.
