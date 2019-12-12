#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=188fQGZIpUvyv-7krCSEqKDUnWhL9uzWP" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=188fQGZIpUvyv-7krCSEqKDUnWhL9uzWP" -o deeplabv3-mobilenetv3-small-voc-500000.tar.gz
tar -zxvf deeplabv3-mobilenetv3-small-voc-500000.tar.gz
rm deeplabv3-mobilenetv3-small-voc-500000.tar.gz

echo Download finished.
