#!/bin/bash

fileid="1UiboYzQQWWUuKAj2al_zQ3u5Bil8H-oa"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o deeplab_mnv3_large_cityscapes_trainfine_769.pb

echo Download finished.
