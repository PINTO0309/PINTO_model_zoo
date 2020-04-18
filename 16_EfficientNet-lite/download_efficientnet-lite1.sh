#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1nqYJeHbGLxsp4WBD_9FVeBBYN0RiFaOX" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1nqYJeHbGLxsp4WBD_9FVeBBYN0RiFaOX" -o efficientnet-lite1.tar.gz

tar -zxvf efficientnet-lite1.tar.gz
rm efficientnet-lite1.tar.gz
echo Download finished.
