#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1PK1GzNVu9slqee_yHJ3v0hV_n3Zc8Lfi" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1PK1GzNVu9slqee_yHJ3v0hV_n3Zc8Lfi" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
