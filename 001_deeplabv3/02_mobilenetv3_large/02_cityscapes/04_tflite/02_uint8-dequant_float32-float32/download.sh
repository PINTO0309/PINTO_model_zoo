#!/bin/bash

fileid="1m_Qy4qtU1zzzYmcJcaDYGyhqX1Q0N8cd"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o deeplab_mnv3_large_cityscapes_trainfine.tflite

echo Download finished.
