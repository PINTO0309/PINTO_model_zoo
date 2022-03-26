#!/bin/bash

fileid="1iLdeN_Ap7v1NSq7W3f2-XT3Gqi_Ev9wj"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o deeplabv3_mnv2_pascal_train_aug_weight_quant.tflite

echo Download finished.
