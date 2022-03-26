#!/bin/bash

fileid="1iaAfhtvKY76URdJ38qNNc9_VR-KpIDnp"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o deeplabv3_mnv2_dm05_pascal_trainaug_weight_quant.tflite

echo Download finished.
