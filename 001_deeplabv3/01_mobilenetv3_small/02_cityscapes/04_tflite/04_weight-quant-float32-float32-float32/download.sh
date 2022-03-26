#!/bin/bash

fileid="1JzYLv7M0KVdLsqwu5lvRFEpUQv68hryk"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o deeplab_mnv3_small_weight_quant_257.tflite

fileid="1JzYLv7M0KVdLsqwu5lvRFEpUQv68hryk"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o deeplab_mnv3_small_weight_quant_769.tflite

echo Download finished.
