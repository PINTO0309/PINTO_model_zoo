#!/bin/bash

fileid="1JDeHbaBk71JkHxkuyhvFjFEJn4CDQ4vr"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ssd_mobilenet_v3_large_coco_full_integer_quant.tflite

echo Download finished.
