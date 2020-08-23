#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1GGBdWFmf7NXW3ek0Zcic1b2aM6da5Q0C" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1GGBdWFmf7NXW3ek0Zcic1b2aM6da5Q0C" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
