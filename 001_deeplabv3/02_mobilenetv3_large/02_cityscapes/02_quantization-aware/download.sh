#!/bin/bash

fileid="147mG4-vOBMPII-mvsMv1bB6g5NAdw9y0"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o deeplabv3-mobilenetv3-large-cityscapes-4361-quant.tar.gz
tar -zxvf deeplabv3-mobilenetv3-large-cityscapes-4361-quant.tar.gz
rm deeplabv3-mobilenetv3-large-cityscapes-4361-quant.tar.gz

echo Download finished.
