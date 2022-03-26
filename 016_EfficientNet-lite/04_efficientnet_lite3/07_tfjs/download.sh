#!/bin/bash

fileid="1V2QSu998wDSP39H5x17t0_h3J2chQ1ns"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o resources.tar.gz

tar -zxvf resources.tar.gz
rm resources.tar.gz
echo Download finished.
