#!/bin/bash

fileid="1AUWcRc18F1As4u34ti9sUcYeZhjUGNIX"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o monodepth2_colormap_depth_weight_quant.tflite

fileid="10apjPIBJcOLvyrwRiFZcRLwZQj8ZgD0a"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o monodepth2_colormap_only_weight_quant.tflite

echo Download finished.

