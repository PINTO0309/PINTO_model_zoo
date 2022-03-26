#!/bin/bash

fileid="1lF7vs43_DJAO1lGLn4LOAZkWbgjKJYax"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o lite-model_efficientdet_lite0_detection_default_1_opt.onnx

echo Download finished.
