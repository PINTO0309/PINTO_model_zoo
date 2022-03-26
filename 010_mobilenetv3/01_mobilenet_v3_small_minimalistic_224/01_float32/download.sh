#!/bin/bash

fileid="1xm5vizWif56d1eBqu81ziJtkgmEujNj0"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o v3-small-minimalistic_224_1.0_float.pb

echo Download finished.
