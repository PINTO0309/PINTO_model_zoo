#!/bin/bash

fileid="1h23k090mcd1sUj_C2w679JM-Ixbza-YI"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ssd_mobilenet_v3_small_coco_integer_quant.tflite

fileid="1ejDQyf6M-i3PPGoSM4SjHHTA7K-Uhc-O"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ssd_mobilenet_v3_small_coco_integer_quant_postprocess.tflite

echo Download finished.
