#!/bin/bash
# yolov3_lite_voc_256_full_integer_quant.tflite
fileid="19Vcthxad-odbcUF5zLF-Y12ADOXG_q0o"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_lite_voc_256_full_integer_quant.tflite
rm cookie

# yolov3_lite_voc_320_full_integer_quant.tflite
fileid="1z90PlRLtZolwqW-3I3_FV2HEljylOIAi"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_lite_voc_320_full_integer_quant.tflite
rm cookie

# yolov3_lite_voc_416_full_integer_quant.tflite
fileid="1QBiU5gGJwV5Jcb488RCqwgVNfvARhY_u"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_lite_voc_416_full_integer_quant.tflite
rm cookie

echo Download finished.
