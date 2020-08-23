#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1iLdeN_Ap7v1NSq7W3f2-XT3Gqi_Ev9wj" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1iLdeN_Ap7v1NSq7W3f2-XT3Gqi_Ev9wj" -o deeplabv3_mnv2_pascal_train_aug_weight_quant.tflite

echo Download finished.
