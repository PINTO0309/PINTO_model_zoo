#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1LGHN27s9gSI8_DP2Bzcf5k9onRD9iwyI" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1LGHN27s9gSI8_DP2Bzcf5k9onRD9iwyI" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
