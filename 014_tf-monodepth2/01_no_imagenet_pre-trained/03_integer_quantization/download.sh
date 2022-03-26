#!/bin/bash

fileid="1aUnn0WVDld8XItk0i5HgKEULPRrHhPQU"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o monodepth2_colormap_depth_integer_quant.tflite

fileid="1jZ9rWmLGTonwnGhjQLO9jPMFjpmR9YG7"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o monodepth2_colormap_only_integer_quant.tflite

echo Download finished.
