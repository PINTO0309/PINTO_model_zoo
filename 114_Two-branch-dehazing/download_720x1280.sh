#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=15XGcvQrkuHEbsd0Vzp4LRmC39blbCSGc" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=15XGcvQrkuHEbsd0Vzp4LRmC39blbCSGc" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
