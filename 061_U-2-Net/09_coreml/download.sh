#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=14y20bM_qFe1hesmaJIV5sdkWnbv_68IM" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=14y20bM_qFe1hesmaJIV5sdkWnbv_68IM" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
