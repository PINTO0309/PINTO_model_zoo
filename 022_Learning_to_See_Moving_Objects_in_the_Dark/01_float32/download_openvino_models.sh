#!/bin/bash

fileid="1goszH1_DlW2qaor8mTMY-f55NgSwxTAS"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o openvino_models.tar.gz
tar -zxvf openvino_models.tar.gz
rm openvino_models.tar.gz
rm cookie

echo Download finished.
