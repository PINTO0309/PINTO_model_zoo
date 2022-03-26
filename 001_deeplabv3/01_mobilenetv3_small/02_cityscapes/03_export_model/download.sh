#!/bin/bash

fileid="1197PaaAUGHAJ_L_fQP5rzPqpCrk62qFM"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o deeplab_mnv3_small_cityscapes_trainfine_769.pb

echo Download finished.
