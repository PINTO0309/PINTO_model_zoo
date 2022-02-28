#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1hCU0Mi99DSquGw_B9Ht1ii8B7JwQ94J5" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1hCU0Mi99DSquGw_B9Ht1ii8B7JwQ94J5" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
