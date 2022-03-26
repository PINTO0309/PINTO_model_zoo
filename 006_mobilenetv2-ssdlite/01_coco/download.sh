#!/bin/bash

fileid="1GAenxYvmPzahzs175Jk_wtt46OW3BiCK"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o anchors.npy

echo Download finished.
