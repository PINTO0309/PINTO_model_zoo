#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1goszH1_DlW2qaor8mTMY-f55NgSwxTAS" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1goszH1_DlW2qaor8mTMY-f55NgSwxTAS" -o openvino_models.tar.gz
tar -zxvf openvino_models.tar.gz
rm openvino_models.tar.gz
echo Download finished.
