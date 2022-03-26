#!/bin/bash

fileid="1UeCXsjP-5vZN_E-7nfaSzo0zue1rD--S"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o monodepth2_colormap_depth_integer_quant.tflite

fileid="1_Qc5GpVJk5JMmOGKBzDpMlej57Hxc5pi"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o monodepth2_colormap_only_integer_quant.tflite

echo Download finished.
