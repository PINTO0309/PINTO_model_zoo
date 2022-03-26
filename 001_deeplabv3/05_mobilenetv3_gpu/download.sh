#!/bin/bash

fileid="1_3zPqzwb85OKGolI2DBBuVvJKPPm4TSE"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o deeplabv3_257_mv_gpu.tflite

echo Download finished.
