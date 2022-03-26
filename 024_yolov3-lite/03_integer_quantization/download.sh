#!/bin/bash
# yolov3_lite_voc_256_integer_quant.tflite
fileid="1DpdWUGbACcTrasPN7U-NkLAtl9IJphV3"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_lite_voc_256_integer_quant.tflite
rm cookie

# yolov3_lite_voc_320_integer_quant.tflite
fileid="1pE8fyw7jU2HxIcbM-lX9p_8dMv03sMLc"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_lite_voc_320_integer_quant.tflite
rm cookie

# yolov3_lite_voc_416_integer_quant.tflite
fileid="1lIHGlslrH3xHx1wBVEnQv1YH88mpVxkd"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_lite_voc_416_integer_quant.tflite
rm cookie

echo Download finished.
