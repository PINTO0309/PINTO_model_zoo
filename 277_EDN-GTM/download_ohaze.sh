#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1LK9FPt0e-yLlUob6pIcNJ_Hux8UbMfjR" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1LK9FPt0e-yLlUob6pIcNJ_Hux8UbMfjR" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
