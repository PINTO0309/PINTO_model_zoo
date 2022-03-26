#!/bin/bash

fileid="18MROBXweoXl7mWVSkq80jSkfiL6dWd6O"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenetv2_fsd2018_41cls_full_integer_quant.tflite

echo Download finished.
