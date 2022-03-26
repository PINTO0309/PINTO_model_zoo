#!/bin/bash

fileid="1xny0Fb_8Qv2_n0xfIVqtSqPuWVndXsXS"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o monodepth2_colormap_depth_float16_quant.tflite

fileid="1t498UjHWXTsDNBxloK8edc0bIimDuPKm"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o monodepth2_colormap_only_float16_quant.tflite

echo Download finished.
