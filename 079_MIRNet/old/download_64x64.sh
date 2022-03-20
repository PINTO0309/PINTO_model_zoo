#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1YV2EzON3_Lr-ypkB9Cj-G1Yjg2cb2jMe" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1YV2EzON3_Lr-ypkB9Cj-G1Yjg2cb2jMe" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
