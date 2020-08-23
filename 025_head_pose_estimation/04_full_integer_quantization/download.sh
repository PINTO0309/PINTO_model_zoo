#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1gLe51rkC3nodvd_PhlUk6B3xx7eM1qt0" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1gLe51rkC3nodvd_PhlUk6B3xx7eM1qt0" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
