#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1hyKPx_MF2_C_bxiw8lQLf6DpJLmMD2X5" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1hyKPx_MF2_C_bxiw8lQLf6DpJLmMD2X5" -o resources.tar.gz

tar -zxvf resources.tar.gz
rm resources.tar.gz
echo Download finished.
