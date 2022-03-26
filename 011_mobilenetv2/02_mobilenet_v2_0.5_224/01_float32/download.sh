#!/bin/bash

fileid="1Plwwp4Oq9d-X643Sn21GpXY5F_irU3lR"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mobilenet_v2_0.5_224_frozen.pb

echo Download finished.
