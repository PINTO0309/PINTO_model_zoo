#!/bin/bash

fileid="1hmzINS-NNDxBQlpMGvAbj837t4d7ob3X"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o bts_densenet161_480_640.tar.gz
tar -zxvf bts_densenet161_480_640.tar.gz
rm bts_densenet161_480_640.tar.gz

echo Download finished.
