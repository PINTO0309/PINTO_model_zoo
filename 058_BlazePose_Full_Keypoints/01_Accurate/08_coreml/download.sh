#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1QbzsM1kBK17J2P-Shcy0quSKFVIfWhId" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1QbzsM1kBK17J2P-Shcy0quSKFVIfWhId" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
