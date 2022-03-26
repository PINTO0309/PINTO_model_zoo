#!/bin/bash

fileid="1P1tpdn9vmwqw4imSYd2cfKX0aO3H9DJX"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o v3-small_224_0.75_float.pb

echo Download finished.
