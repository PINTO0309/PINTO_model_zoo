#!/bin/bash

fileid="1X_YS6Mr3Lmo9sLfCbVHa-RaKlni9FxF7"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o style_predict_quantized_256.tflite

fileid="1mGwTGI1bcuTe4fUFGBvwr9AMGN0HcOzn"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o style_transfer_quantized_dynamic.tflite

echo Download finished.
