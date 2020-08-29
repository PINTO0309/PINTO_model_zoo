#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=11t2GBX9F-Zk48xjBRbBr-rzy2htVmBok" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=11t2GBX9F-Zk48xjBRbBr-rzy2htVmBok" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
