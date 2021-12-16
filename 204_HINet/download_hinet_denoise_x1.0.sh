#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1p16Gx7Ekz2jWmX2uj-VGjbD3gOAaaheU" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1p16Gx7Ekz2jWmX2uj-VGjbD3gOAaaheU" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
