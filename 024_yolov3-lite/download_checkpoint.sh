#!/bin/bash

fileid="1f0OOcM1g-v5WMBtwTvGfsQ5u5eUdyXht"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o checkpoint.tar.gz
tar -zxvf checkpoint.tar.gz
rm checkpoint.tar.gz
rm cookie

echo Download finished.
