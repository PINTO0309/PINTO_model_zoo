#!/bin/bash

fileid="1NgaBdkqoT8gjiVjMf6hHZqzEK2Es-e9w"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v2_1.0_224_frozen.pb

echo Download finished.
