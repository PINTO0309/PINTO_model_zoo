#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1XIHKvjKfA18-X2TiMmX3zXstGhY1V4-q" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1XIHKvjKfA18-X2TiMmX3zXstGhY1V4-q" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
