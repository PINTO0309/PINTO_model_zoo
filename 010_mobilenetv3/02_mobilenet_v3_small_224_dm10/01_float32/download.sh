#!/bin/bash

fileid="1pLHMKmWdO3PuIF1so9SRhZ9Wdjr3gua-"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o v3-small_224_1.0_float.pb

echo Download finished.
