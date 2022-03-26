#!/bin/bash

fileid="1LjTqn5nChAVKhXgwBUp00XIKXoZrs9sB"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ssdlite_mobilenet_v2_coco_300_integer_quant.tflite

fileid="1poH1bnh_4UYbYoNtunDMJPvLMMZng4VQ"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ssdlite_mobilenet_v2_coco_300_integer_quant_with_postprocess.tflite

echo Download finished.
