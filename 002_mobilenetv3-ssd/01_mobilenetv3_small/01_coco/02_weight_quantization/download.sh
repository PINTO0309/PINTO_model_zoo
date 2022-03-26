#!/bin/bash

fileid="1qWiKax3QMvPOtZme-Aqglw_nRxQ1cpAX"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ssd_mobilenet_v3_small_coco_weight_quant.tflite

echo Download finished.
