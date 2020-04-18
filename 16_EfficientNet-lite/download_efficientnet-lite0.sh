#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1M53Yoa5Z_kBdapi78GCWBDZilFYUFuex" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1M53Yoa5Z_kBdapi78GCWBDZilFYUFuex" -o efficientnet-lite0.tar.gz

tar -zxvf efficientnet-lite0.tar.gz
rm efficientnet-lite0.tar.gz
echo Download finished.
