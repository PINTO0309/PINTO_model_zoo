#!/bin/bash
# yolov3_nano_voc_256_float16_quant.tflite
fileid="1E-19_wtOg3El6cuXiaB1iy8Dvvs80NOo"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_nano_voc_256_float16_quant.tflite
rm cookie

# yolov3_nano_voc_320_float16_quant.tflite
fileid="1Xmu4i99ZNGCWCpVmOhcRvxRQDMYDu6h0"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_nano_voc_320_float16_quant.tflite
rm cookie

# yolov3_nano_voc_416_float16_quant.tflite
fileid="1tqvFG72dkdrcpKEIFv-ipoojmpmWfvzm"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o yolov3_nano_voc_416_float16_quant.tflite
rm cookie

echo Download finished.
