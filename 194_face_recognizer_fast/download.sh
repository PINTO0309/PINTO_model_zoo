#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1U3AgUDev2Bpp0JVK9qvF2U-SKKQ5wypK" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1U3AgUDev2Bpp0JVK9qvF2U-SKKQ5wypK" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
