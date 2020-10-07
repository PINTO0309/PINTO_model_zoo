#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=14m1oO0KiucYXTLLiQ10XBTIcU8zHU98A" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=14m1oO0KiucYXTLLiQ10XBTIcU8zHU98A" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
