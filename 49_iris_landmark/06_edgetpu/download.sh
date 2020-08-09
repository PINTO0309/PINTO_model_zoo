#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1eFD3DcWx4hei1XP_E50dWT_FTrSd0hjL" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1eFD3DcWx4hei1XP_E50dWT_FTrSd0hjL" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
