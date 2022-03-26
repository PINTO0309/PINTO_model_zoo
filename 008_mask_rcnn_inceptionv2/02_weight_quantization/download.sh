#!/bin/bash

fileid="1nE53eJt5mBOmzinGxHCR5eIcbghR80Ke"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o mask_rcnn_inception_v2_coco_256_weight_quant.tflite

echo Download finished.
