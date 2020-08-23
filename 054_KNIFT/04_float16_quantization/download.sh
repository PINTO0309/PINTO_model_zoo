#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1hkylsrdInHj2XOCV_tf7k8CUwKPm-81B" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1hkylsrdInHj2XOCV_tf7k8CUwKPm-81B" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
