#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1BCKCewNLLE04lyaZaPi942nvjZYkhCvg" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1BCKCewNLLE04lyaZaPi942nvjZYkhCvg" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
